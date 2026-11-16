# t01_timing_cpu: fixture_timing (performance)

**Status: PASS**


- commit `59c68974b553c9ddd84ef42767ca866888f5299e` (tracked tree dirty: False), torch 2.14.1+cu126, transformers 4.57.6, peft 0.21.3, GPU None
- config `research/l2l/configs/resident.yaml` sha256 `bbd460751d39cdb7`; fields used: ['model.dtype', 'execution']; backend `resident`
- input sha256 `cb7569af732767ae`, tolerances sha256 `b722e3c70069a4f8`

| side | n | median s | IQR s | median 95% CI s |
|---|---|---|---|---|
| reference | 20 | 1.490e-02 | [0.012354886246612296, 0.015867082749537076] | [0.013164691499696346, 0.01559664649903425] |
| candidate | 20 | 1.447e-02 | [0.012725626253086375, 0.014869712498693843] | [0.013351540503208525, 0.014836376496532466] |

candidate / reference median ratio 0.972, 95% CI [0.8864657406936042, 1.1181758416852914]: no detectable difference

memory snapshot: `{"kind": "snapshot", "cpu": {"rss_mib": 704.56640625, "peak_rss_mib": 704.56640625}, "gpu": null}`
