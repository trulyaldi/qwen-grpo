# t01_timing_cuda: fixture_timing (performance)

**Status: PASS**


- commit `59c68974b553c9ddd84ef42767ca866888f5299e` (tracked tree dirty: False), torch 2.14.1+cu126, transformers 4.57.6, peft 0.21.3, GPU NVIDIA GeForce RTX 3050 Ti Laptop GPU
- config `research/l2l/configs/resident.yaml` sha256 `bbd460751d39cdb7`; fields used: ['model.dtype', 'execution']; backend `resident`
- input sha256 `cb7569af732767ae`, tolerances sha256 `b722e3c70069a4f8`

| side | n | median s | IQR s | median 95% CI s |
|---|---|---|---|---|
| reference | 20 | 2.294e-02 | [0.02230915975087555, 0.02370902874827152] | [0.022380864498700248, 0.0233677694996004] |
| candidate | 20 | 2.266e-02 | [0.021928509000645136, 0.024313424497449887] | [0.022056737998354947, 0.024033539499214385] |

candidate / reference median ratio 0.988, 95% CI [0.9505620011633801, 1.0537286284421112]: no detectable difference

memory snapshot: `{"kind": "snapshot", "cpu": {"rss_mib": 1447.53125, "peak_rss_mib": 1447.53125}, "gpu": {"allocated_mib": 16.720703125, "reserved_mib": 22.0, "peak_allocated_mib": 17.15185546875, "peak_reserved_mib": 22.0, "device_used_mib_mem_get_info": 857.3125, "nvidia_smi_compute_apps_raw": "227238, [N/A]", "nvidia_smi_memory_used_raw": "134", "process_used_mib_nvidia_smi": null, "device_used_mib_nvidia_smi": 134.0}, "pinned_host": {"allocated_bytes.allocated": 1, "allocated_bytes.current": 1, "allocated_bytes.freed": 0, "allocated_bytes.peak": 1}}`
