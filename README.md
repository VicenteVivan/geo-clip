<div align="center">    
 
# 🌎 GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://arxiv.org/abs/2309.16020v2)
[![Conference](https://img.shields.io/badge/NeurIPS-2023-blue)]()

![ALT TEXT](https://i.ibb.co/fDKYTQY/Screenshot-2023-12-12-at-12-01-16-PM-modified.png)

</div>
 
## Description
GeoCLIP addresses the challenges of worldwide image geo-localization by introducing a novel CLIP-inspired approach that aligns images with geographical locations, achieving state-of-the-art results on geo-localization and GPS to vector representation on benchmark datasets (Im2GPS3k, YFCC26k, GWS15k, and Geo-Tagged NUS-Wide Dataset). Our location encoder models the Earth as a continuous function, learning semantically rich, CLIP-aligned features that are suitable for geo-localization. Additionally, our location encoder architecture generalizes, making it suitable for use as a GPS encoder that can be used to aid geo-aware neural architectures.

_🚧 Repo Under Construction 🔨_

## Usage: Pre-Trained Location Encoder

```python
import torch
import torch.nn as nn
from location_endoder import LocationEncoder

gps_encoder = LocationEncoder()
gps_encoder.load_state_dict(torch.load('location_encoder_weights.pth'))

loc_data = torch.Tensor([[40.7128, -74.0060], [34.0522, -118.2437]])  # NYC and LA in lat, long
output = gps_encoder(loc_data)
print(output.shape)
```

### Citation

```
@article{cepeda2023geoclip,
  title={GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization},
  author={Vivanco, Vicente and Nayak, Gaurav Kumar and Shah, Mubarak},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```
