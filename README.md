# SC-Gen-3

This repository contains utilities for analysing Companies House documents.

## Parallel OCR

The `CompanyHouseDocumentPipeline` class and the `run_batch_company_analysis` function now accept an `ocr_workers` parameter. When set to a value greater than one, text extraction of downloaded documents will use a thread pool to run OCR in parallel via `_process_documents`.

```python
pipeline = CompanyHouseDocumentPipeline("01234567", ch_api_key="APIKEY")
result = pipeline.run(ocr_workers=4)
```

Sequential behaviour is preserved by leaving `ocr_workers` at its default of `1`.

