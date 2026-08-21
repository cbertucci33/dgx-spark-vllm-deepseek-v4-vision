# Hugging Face publication checklist

## Technical gates

- [x] Source repositories and immutable revisions recorded in `SOURCE_PINS.json`.
- [x] Tower, projector, runtime wheel, image, and source revisions pinned by SHA-256 or commit digest.
- [x] Model card explains composition rather than claiming tensor averaging.
- [x] All 48 language shards, the overlay shard, and the weight index are bound by the pinned `LANGUAGE_WEIGHTS.sha256` manifest.
- [x] The assembler rejects shard substitution, index mutation, unexpected vision/runtime files, symlinks, traversal, and overlapping trees.
- [x] Hugging Face LFS patterns cover `.safetensors` and `.pt` files.
- [x] Final 102-entry repository manifests are identical and verified on both physical model copies.
- [x] Apache-2.0 is the top-level model license and inherited MIT/Apache notices are bundled.
- [x] The model card passes Hugging Face server-side metadata validation and strict citation validation.
- [x] Text and two-image OCR smoke tests passed on the qualified runtime.

## Publication gates

- [x] Record FlyCockpit's Apache-2.0 repository tag and its upstream-terms qualification for the tower.
- [x] Use `cbert33/DeepSeek-V4-Flash-0731-abliterated-vision` as the model repository.
- [x] Add explicit uncensored-model responsibility and intended-use language.
- [x] Publish the complete 168+ GB assembled model privately for final review.
- [ ] Obtain explicit approval immediately before changing the Hub repository to public.

## Staged upload command — do not run before approval

```bash
hf upload OWNER/REPOSITORY \
  /path/to/DeepSeek-V4-Flash-0731-abliterated-vision \
  --repo-type model
```

Use `hf upload` so large-file transfers are resumable. Verify the remote commit and file list after upload before announcing success.
