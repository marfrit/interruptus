// interruptus-extract: residual-stream activation extractor (milestone 1)
//
// Modes (env-selected, so we never touch the arg parser):
//   IEX_OBSERVE=1        -> Stage A: log every graph tensor (name/op/shape/type/host), no data copy.
//   IEX_BATCH=<file>     -> BATCH extract: one model load, many prompts. See format below.
//   otherwise            -> single-prompt extract from -p.
//
// Extract env:
//   IEX_OUTDIR=<dir>     output directory (required in extract/batch mode)
//   IEX_PREFIX=<str>     residual tensor name prefix (default "l_out-")
//   IEX_LAYERS=a,b,c     explicit target layers; else round(0.60*n_layer)..round(0.75*n_layer)
//
// BATCH file format (binary-safe; prompts may contain newlines/quotes):
//   <id>\t<byte_len>\n
//   <byte_len bytes of prompt>\n
//   ... repeated ...
// For each record: KV is cleared, the prompt is tokenized (add_bos per model, parse_special=true),
// decoded (prefill only), and the L<il> last-token vectors are written concatenated (sorted layer
// order) to <outdir>/<id>.f32 . A manifest line is appended to <outdir>/manifest.tsv:
//   <id>\t<n_tokens>\t<last_token_id>\t<layers_csv>

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <set>
#include <map>

struct extract_ctx {
    bool                     observe = false;
    int                      n_embd  = 0;
    std::string              prefix  = "l_out-";
    std::set<int>            target_layers;
    std::string              outdir;
    std::map<int, std::vector<float>> vecs;   // il -> last-token vector (n_embd f32)
    std::vector<uint8_t>     scratch;
};

static std::string ne_str(const ggml_tensor * t) {
    std::string s;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        s += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS) s += ",";
    }
    return s;
}

static int layer_from_name(const char * name, const std::string & prefix) {
    size_t pl = prefix.size();
    if (strncmp(name, prefix.c_str(), pl) != 0) return -1;
    const char * p = name + pl;
    if (!*p) return -1;
    char * end = nullptr;
    long v = strtol(p, &end, 10);
    if (end == p || *end != '\0') return -1;
    return (int) v;
}

static bool cb_eval(ggml_tensor * t, bool ask, void * ud) {
    auto * e = (extract_ctx *) ud;
    if (ask) return true;

    if (e->observe) {
        fprintf(stderr, "IEX %-28s op=%-14s ne=[%s] type=%s host=%d\n",
                t->name, ggml_op_desc(t), ne_str(t).c_str(),
                ggml_type_name(t->type), ggml_backend_buffer_is_host(t->buffer) ? 1 : 0);
        return true;
    }

    int il = layer_from_name(t->name, e->prefix);
    if (il < 0 || e->target_layers.find(il) == e->target_layers.end()) return true;
    if (t->type != GGML_TYPE_F32) return true;

    const int64_t ne0 = t->ne[0];   // n_embd
    const int64_t ne1 = t->ne[1];   // n_tokens
    const int64_t col = ne1 - 1;    // last prompt token

    std::vector<float> & out = e->vecs[il];
    out.resize(ne0);

    const bool host = ggml_backend_buffer_is_host(t->buffer);
    const uint8_t * base;
    if (host) {
        base = (const uint8_t *) t->data;
    } else {
        size_t nbytes = ggml_nbytes(t);
        e->scratch.resize(nbytes);
        ggml_backend_tensor_get(t, e->scratch.data(), 0, nbytes);
        base = e->scratch.data();
    }
    for (int64_t i = 0; i < ne0; ++i) {
        out[i] = *(const float *) (base + col * t->nb[1] + i * t->nb[0]);
    }
    return true;
}

static std::string json_escape(const std::string & s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else o += c;
    }
    return o;
}

// decode a single prompt string; returns n_tokens (>0) on success, -1 on failure.
// last_tok receives the id of the final prompt token.
static int decode_prompt(llama_context * ctx, const std::string & prompt, int n_batch, int & last_tok) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, prompt, add_bos, true);
    if (tokens.empty()) return -1;
    if ((int) tokens.size() > n_batch) {
        LOG_ERR("IEX: prompt has %zu tokens > n_batch %d -- skipping\n", tokens.size(), n_batch);
        return -1;
    }
    last_tok = tokens.back();
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) return -1;
    return (int) tokens.size();
}

static void write_vectors(extract_ctx & ex, const std::string & id, std::vector<int> & layers_written) {
    char path[2048];
    snprintf(path, sizeof(path), "%s/%s.f32", ex.outdir.c_str(), id.c_str());
    FILE * f = fopen(path, "wb");
    if (!f) { LOG_ERR("IEX: cannot open %s\n", path); return; }
    for (int il : ex.target_layers) {           // sorted (std::set) 24..30
        auto it = ex.vecs.find(il);
        if (it == ex.vecs.end()) continue;
        fwrite(it->second.data(), sizeof(float), it->second.size(), f);
        layers_written.push_back(il);
    }
    fclose(f);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    extract_ctx ex;
    ex.observe = getenv("IEX_OBSERVE") != nullptr;
    if (const char * p = getenv("IEX_PREFIX")) ex.prefix = p;
    if (const char * o = getenv("IEX_OUTDIR")) ex.outdir = o;
    const char * batch_file = getenv("IEX_BATCH");

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = cb_eval;
    params.cb_eval_user_data = &ex;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) { LOG_ERR("IEX: failed to init\n"); return 1; }

    const int n_layer = llama_model_n_layer(model);
    const int n_embd  = llama_model_n_embd(model);
    const int n_batch = (int) llama_n_batch(ctx);
    ex.n_embd = n_embd;

    if (const char * l = getenv("IEX_LAYERS")) {
        std::string s = l; size_t pos = 0;
        while (pos < s.size()) {
            size_t comma = s.find(',', pos);
            std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            if (!tok.empty()) ex.target_layers.insert(atoi(tok.c_str()));
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
    } else {
        int lo = (int) std::lround(0.60 * n_layer);
        int hi = (int) std::lround(0.75 * n_layer);
        for (int il = lo; il <= hi; ++il) ex.target_layers.insert(il);
    }

    LOG_INF("IEX: n_layer=%d n_embd=%d n_batch=%d observe=%d prefix='%s'\n",
            n_layer, n_embd, n_batch, ex.observe ? 1 : 0, ex.prefix.c_str());
    {
        std::string ls; for (int il : ex.target_layers) { ls += std::to_string(il); ls += " "; }
        LOG_INF("IEX: target_layers = %s\n", ls.c_str());
    }

    // ---------------- BATCH MODE ----------------
    if (batch_file && !ex.observe) {
        if (ex.outdir.empty()) { LOG_ERR("IEX: IEX_OUTDIR required\n"); return 1; }

        FILE * bf = fopen(batch_file, "rb");
        if (!bf) { LOG_ERR("IEX: cannot open batch file %s\n", batch_file); return 1; }
        fseek(bf, 0, SEEK_END); long sz = ftell(bf); fseek(bf, 0, SEEK_SET);
        std::string buf; buf.resize(sz);
        if (fread(&buf[0], 1, sz, bf) != (size_t) sz) { LOG_ERR("IEX: read error\n"); return 1; }
        fclose(bf);

        char mpath[2048];
        snprintf(mpath, sizeof(mpath), "%s/manifest.tsv", ex.outdir.c_str());
        FILE * man = fopen(mpath, "w");
        fprintf(man, "id\tn_tokens\tlast_token_id\tlayers\n");

        size_t pos = 0; int done = 0, failed = 0;
        while (pos < buf.size()) {
            // header line: <id>\t<len>\n
            size_t nl = buf.find('\n', pos);
            if (nl == std::string::npos) break;
            std::string hdr = buf.substr(pos, nl - pos);
            pos = nl + 1;
            size_t tab = hdr.find('\t');
            if (tab == std::string::npos) { LOG_ERR("IEX: bad header '%s'\n", hdr.c_str()); break; }
            std::string id = hdr.substr(0, tab);
            long plen = atol(hdr.c_str() + tab + 1);
            if (pos + (size_t) plen > buf.size()) { LOG_ERR("IEX: truncated record %s\n", id.c_str()); break; }
            std::string prompt = buf.substr(pos, plen);
            pos += plen;
            if (pos < buf.size() && buf[pos] == '\n') pos += 1;   // trailing newline

            llama_memory_clear(llama_get_memory(ctx), true);
            ex.vecs.clear();
            int last_tok = -1;
            int nt = decode_prompt(ctx, prompt, n_batch, last_tok);
            if (nt < 0) { failed++; LOG_ERR("IEX: decode failed for %s\n", id.c_str()); continue; }

            std::vector<int> lw;
            write_vectors(ex, id, lw);
            std::string lcsv; for (size_t i = 0; i < lw.size(); ++i) { if (i) lcsv += ","; lcsv += std::to_string(lw[i]); }
            fprintf(man, "%s\t%d\t%d\t%s\n", id.c_str(), nt, last_tok, lcsv.c_str());
            fflush(man);
            done++;
            if (done % 25 == 0) LOG_INF("IEX: batch progress %d done, %d failed\n", done, failed);
        }
        fclose(man);
        LOG_INF("IEX: BATCH COMPLETE: %d done, %d failed, manifest=%s\n", done, failed, mpath);
        llama_backend_free();
        return 0;
    }

    // ---------------- OBSERVE / SINGLE MODE ----------------
    if (!ex.observe && ex.outdir.empty()) { LOG_ERR("IEX: IEX_OUTDIR required in extract mode\n"); return 1; }

    int last_tok = -1;
    int nt = decode_prompt(ctx, params.prompt, n_batch, last_tok);
    if (nt < 0) { LOG_ERR("IEX: decode failed\n"); return 1; }
    LOG_INF("IEX: n_input_tokens = %d last_token_id = %d\n", nt, last_tok);

    if (!ex.observe) {
        std::vector<int> lw;
        write_vectors(ex, "single", lw);
        char mpath[2048];
        snprintf(mpath, sizeof(mpath), "%s/meta.json", ex.outdir.c_str());
        FILE * mf = fopen(mpath, "w");
        if (mf) {
            fprintf(mf, "{\n");
            fprintf(mf, "  \"model_path\": \"%s\",\n", json_escape(params.model.path).c_str());
            fprintf(mf, "  \"n_layer\": %d,\n  \"n_embd\": %d,\n  \"n_tokens\": %d,\n", n_layer, n_embd, nt);
            fprintf(mf, "  \"prefix\": \"%s\",\n", ex.prefix.c_str());
            fprintf(mf, "  \"prompt\": \"%s\",\n", json_escape(params.prompt).c_str());
            fprintf(mf, "  \"target_layers\": [");
            bool first = true; for (int il : ex.target_layers) { if (!first) fprintf(mf, ", "); fprintf(mf, "%d", il); first = false; }
            fprintf(mf, "]\n}\n");
            fclose(mf);
        }
    }

    llama_backend_free();
    return 0;
}
