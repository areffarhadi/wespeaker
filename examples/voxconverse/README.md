This is a **WeSpeaker** speaker diarization recipe on the Voxconverse 2020 dataset. It focused on a ``in the wild`` scenario, which was collected from YouTube videos with a semi-automatic pipeline and released for the diarization track in VoxSRC 2020 Challenge. See https://www.robots.ox.ac.uk/~vgg/data/voxconverse/ for more detailed information.

Two recipes are provided, including **v1** and **v2**. Their only difference is that in **v2**, we split the Fbank extraction, embedding extraction and clustering modules to different stages. We recommend newcomers to follow the **v2** recipe and run it stage by stage.

🔥 UPDATE 2024.08.20:
* silero-vad v5.1 is used in place of v3.1
* umap dimensionality reduction + hdbscan clustering is also supported in v2

### Reported DER (v2, test partition)

md-eval **OVERALL SPEAKER DIARIZATION ERROR** (percent of scored speaker time, ALL):

| Pipeline | Script | DER (%) |
|----------|--------|--------:|
| WPT + zl389 Adapter/ASP/Bottleneck + w2v-BERT-2.0 (USM v2 GPU mel checkpoint) | [`v2/run_w2vbert_wpt_zl389_adapter_gpu_mel.sh`](v2/run_w2vbert_wpt_zl389_adapter_gpu_mel.sh) | **5.74** |

Details from the same evaluation (speaker-weighted confusion summary): MISS 14.8% (count) / 2.8% (time); FALSE ALARM 3.5% (count) / 1.1% (time).
