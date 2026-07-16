# Data Source Registry

## Purpose

This registry tracks all data sources used in the project, including their licensing, acquisition method, and usage permissions.

## Schema

Each data source entry must include:

| Field | Type | Description |
|-------|------|-------------|
| `source_id` | string | Unique stable identifier |
| `source_name` | string | Human-readable name |
| `source_url` | string | Original source URL or dataset card |
| `publisher` | string | Author/organization |
| `acquired_at` | date | Download/acquisition date (ISO 8601) |
| `published_at` | date | Original publication date |
| `license` | string | SPDX identifier or description |
| `allowed_uses` | dict | Per-use-type boolean flags |
| `contains_pii` | boolean | Whether data may contain PII |
| `checksum_sha256` | string | SHA-256 hash of original file |
| `version` | string | Dataset revision or document version |
| `notes` | string | Additional restrictions or notes |

## allowed_uses Flags

```json
{
  "train": false,       // Can be used in training
  "evaluate": true,     // Can be used in evaluation
  "local_demo": true,   // Can be used in local demos
  "redistribute": false // Can be redistributed
}
```

## Example Entry

```json
{
  "source_id": "cninfo_annual_002594_2025",
  "source_name": "示例公司2025年年度报告",
  "source_url": "https://www.cninfo.com.cn/new/disclosure/detail?stockCode=002594&announcementId=123456",
  "publisher": "示例股份有限公司",
  "acquired_at": "2026-03-28",
  "published_at": "2026-03-28",
  "license": "official-disclosure",
  "allowed_uses": {
    "train": false,
    "evaluate": true,
    "local_demo": true,
    "redistribute": false
  },
  "contains_pii": false,
  "checksum_sha256": "abc123...",
  "version": "1.0",
  "notes": "Official stock exchange disclosure. For local research only. PDF redistribution not permitted."
}
```

## Registry File

The actual registry is stored in `data/registry/sources.csv` and updated with each new data acquisition.

## Workflow

1. Before downloading any data, create a registry entry
2. Fill in known information before acquisition
3. Complete entry after download (checksum, license verification)
4. Move to quarantine if license is unclear
5. Never use data without a complete registry entry
