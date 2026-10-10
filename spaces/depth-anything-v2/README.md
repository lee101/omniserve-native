---
title: Depth Anything V2 API
emoji: 📏
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
license: apache-2.0
---

# Depth Anything V2 API

Commercially deployable Depth Anything V2 Small inference API.

`POST /v1/depth-estimations`

```json
{"image_url":"https://example.com/image.jpg","output_format":"png16","invert":true,"preview":true}
```
