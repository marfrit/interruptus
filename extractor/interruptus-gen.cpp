// interruptus-gen: generation-loop residual extractor (milestone 1, part 2 / commitment-index).
//
// Unlike interruptus-extract (prefill-only, last prompt token), this tool OWNS the generation
// loop: prefill the prompt, then sample->decode token by token (KV cache persists), and at EACH
// generated token copy the residual `l_out-<il>` last-position vector. Stop at EOS or a token cap.
//
// Env:
//   IEX_OUTDIR=<dir>     output directory (required)
//   IEX_BATCH=<file>     batch file: <id>\t<byte_len>\n<byte_len prompt bytes>\n ... (repeated)
//   IEX_LAYERS=a,b,c     target layers (default 29). Vectors written concatenated (sorted) per token.
//   IEX_PREFIX=<str>     residual tensor name prefix (default "l_out-")
//   IEX_MAXGEN=<n>       max generated tokens (default 1024)
//
// Sampling comes from the standard llama.cpp arg parser (common_params_sampling): pass
//   --temp 0.6 --top-p 0.95 --top-k 20 --seed 0   on the command line.
//
// Per record writes:
//   <outdir>/<id>.gen.f32   n_gen * (n_layers*n_embd) float32, per-token residual vectors
//   <outdir>/<id>.gen.txt   the generated text (thinking + answer)
// And a manifest line to <outdir>/gen_manifest.tsv:
//   <id>\t<n_prompt>\t<n_gen>\t<stop_reason>\t<layers_csv>

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <set>
#include <map>

struct extract_ctx {
    int                      n_embd  = 0;
    std::string              prefix  = "l_out-";
    std::set<int>            target_layers;
    std::map<int, std::vector<float>> vecs;   // il -> last-token vector (n_embd f32)
    std::vector<uint8_t>     scratch;
};

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

// capture the LAST-position residual of every target layer on every graph eval.
static bool cb_eval(ggml_tensor * t, bool ask, void * ud) {
    auto * e = (extract_ctx *) ud;
    if (ask) return true;

    int il = layer_from_name(t->name, e->prefix);
    if (il < 0 || e->target_layers.find(il) == e->target_layers.end()) return true;
    if (t->type != GGML_TYPE_F32) return true;

    const int64_t ne0 = t->ne[0];   // n_embd
    const int64_t ne1 = t->ne[1];   // n_tokens in this batch
    const int64_t col = ne1 - 1;    // last token of the batch

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

// run generation for one prompt; returns n_gen (>=0) or -1 on prefill failure.
static int gen_one(llama_context * ctx, common_sampler * smpl, extract_ctx & ex,
                   const std::string & prompt, int n_batch, int maxgen,
                   std::vector<float> & run_vecs, std::string & gentext,
                   std::string & stop_reason) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    llama_memory_clear(llama_get_memory(ctx), true);
    common_sampler_reset(smpl);

    std::vector<llama_token> tokens = common_tokenize(ctx, prompt, add_bos, true);
    if (tokens.empty()) return -1;
    if ((int) tokens.size() > n_batch) {
        LOG_ERR("IEX-GEN: prompt %zu tokens > n_batch %d -- skipping\n", tokens.size(), n_batch);
        return -1;
    }
    const int n_prompt = (int) tokens.size();

    // prefill (residual of last prompt token is captured but ignored)
    ex.vecs.clear();
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) return -1;

    const size_t per_tok = ex.target_layers.size() * (size_t) ex.n_embd;
    run_vecs.clear();
    gentext.clear();
    stop_reason = "cap";

    int n_gen = 0;
    for (; n_gen < maxgen; ++n_gen) {
        llama_token id = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, id, true);
        if (llama_vocab_is_eog(vocab, id)) { stop_reason = "eog"; break; }

        gentext += common_token_to_piece(ctx, id);

        // decode this generated token; cb_eval captures its residual at the last (only) position
        ex.vecs.clear();
        if (llama_decode(ctx, llama_batch_get_one(&id, 1))) { stop_reason = "decode_err"; break; }

        // append target-layer vectors (sorted layer order) for this token
        run_vecs.reserve(run_vecs.size() + per_tok);
        for (int il : ex.target_layers) {
            auto it = ex.vecs.find(il);
            if (it == ex.vecs.end() || (int) it->second.size() != ex.n_embd) {
                // missing capture -> pad zeros so row count stays consistent
                run_vecs.insert(run_vecs.end(), ex.n_embd, 0.0f);
            } else {
                run_vecs.insert(run_vecs.end(), it->second.begin(), it->second.end());
            }
        }
    }
    (void) n_prompt;
    return n_gen;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    extract_ctx ex;
    if (const char * p = getenv("IEX_PREFIX")) ex.prefix = p;
    std::string outdir;
    if (const char * o = getenv("IEX_OUTDIR")) outdir = o;
    const char * batch_file = getenv("IEX_BATCH");
    int maxgen = 1024;
    if (const char * m = getenv("IEX_MAXGEN")) maxgen = atoi(m);

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
    if (model == nullptr || ctx == nullptr) { LOG_ERR("IEX-GEN: failed to init\n"); return 1; }

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
        ex.target_layers.insert(29);
    }

    LOG_INF("IEX-GEN: n_layer=%d n_embd=%d n_batch=%d maxgen=%d prefix='%s'\n",
            n_layer, n_embd, n_batch, maxgen, ex.prefix.c_str());
    {
        std::string ls; for (int il : ex.target_layers) { ls += std::to_string(il); ls += " "; }
        LOG_INF("IEX-GEN: target_layers = %s\n", ls.c_str());
    }

    if (outdir.empty()) { LOG_ERR("IEX-GEN: IEX_OUTDIR required\n"); return 1; }
    if (!batch_file)    { LOG_ERR("IEX-GEN: IEX_BATCH required\n"); return 1; }

    common_sampler * smpl = common_sampler_init(model, params.sampling);
    if (!smpl) { LOG_ERR("IEX-GEN: sampler init failed\n"); return 1; }

    FILE * bf = fopen(batch_file, "rb");
    if (!bf) { LOG_ERR("IEX-GEN: cannot open batch file %s\n", batch_file); return 1; }
    fseek(bf, 0, SEEK_END); long sz = ftell(bf); fseek(bf, 0, SEEK_SET);
    std::string buf; buf.resize(sz);
    if (fread(&buf[0], 1, sz, bf) != (size_t) sz) { LOG_ERR("IEX-GEN: read error\n"); return 1; }
    fclose(bf);

    char mpath[2048];
    snprintf(mpath, sizeof(mpath), "%s/gen_manifest.tsv", outdir.c_str());
    FILE * man = fopen(mpath, "w");
    fprintf(man, "id\tn_prompt\tn_gen\tstop_reason\tlayers\n");

    std::string lcsv; { bool f=true; for (int il : ex.target_layers) { if(!f) lcsv+=","; lcsv+=std::to_string(il); f=false; } }

    size_t pos = 0; int done = 0, failed = 0;
    while (pos < buf.size()) {
        size_t nl = buf.find('\n', pos);
        if (nl == std::string::npos) break;
        std::string hdr = buf.substr(pos, nl - pos);
        pos = nl + 1;
        size_t tab = hdr.find('\t');
        if (tab == std::string::npos) { LOG_ERR("IEX-GEN: bad header '%s'\n", hdr.c_str()); break; }
        std::string id = hdr.substr(0, tab);
        long plen = atol(hdr.c_str() + tab + 1);
        if (pos + (size_t) plen > buf.size()) { LOG_ERR("IEX-GEN: truncated record %s\n", id.c_str()); break; }
        std::string prompt = buf.substr(pos, plen);
        pos += plen;
        if (pos < buf.size() && buf[pos] == '\n') pos += 1;

        // count prompt tokens for the manifest
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const bool add_bos = llama_vocab_get_add_bos(vocab);
        int n_prompt = (int) common_tokenize(ctx, prompt, add_bos, true).size();

        std::vector<float> run_vecs;
        std::string gentext, stop_reason;
        int n_gen = gen_one(ctx, smpl, ex, prompt, n_batch, maxgen, run_vecs, gentext, stop_reason);
        if (n_gen < 0) { failed++; LOG_ERR("IEX-GEN: gen failed for %s\n", id.c_str()); continue; }

        char vpath[2048];
        snprintf(vpath, sizeof(vpath), "%s/%s.gen.f32", outdir.c_str(), id.c_str());
        FILE * vf = fopen(vpath, "wb");
        if (vf) { fwrite(run_vecs.data(), sizeof(float), run_vecs.size(), vf); fclose(vf); }

        char tpath[2048];
        snprintf(tpath, sizeof(tpath), "%s/%s.gen.txt", outdir.c_str(), id.c_str());
        FILE * tf = fopen(tpath, "wb");
        if (tf) { fwrite(gentext.data(), 1, gentext.size(), tf); fclose(tf); }

        fprintf(man, "%s\t%d\t%d\t%s\t%s\n", id.c_str(), n_prompt, n_gen, stop_reason.c_str(), lcsv.c_str());
        fflush(man);
        done++;
        LOG_INF("IEX-GEN: %s n_prompt=%d n_gen=%d stop=%s\n", id.c_str(), n_prompt, n_gen, stop_reason.c_str());
    }
    fclose(man);
    LOG_INF("IEX-GEN: BATCH COMPLETE: %d done, %d failed, manifest=%s\n", done, failed, mpath);

    common_sampler_free(smpl);
    llama_backend_free();
    return 0;
}
