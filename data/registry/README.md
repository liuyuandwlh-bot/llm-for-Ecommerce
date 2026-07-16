# Data Registry

This directory tracks data sources used in the project.

## Schema

| Field | Type | Description |
|--------|------|-------------|
| source_id | string | Unique identifier |
| source_name | string | Human readable name |
| source_url | string | URL or location |
| publisher | string | Data publisher |
| acquired_at | date | Acquisition date |
| published_at | date | Publication date |
| license | string | License name |
| allowed_train | bool | Allowed for training |
| allowed_evaluate | bool | Allowed for evaluation |
| allowed_local_demo | bool | Allowed for local demo |
| allowed_redistribute | bool | Allowed for redistribution |
| contains_pii | bool | Contains PII |
| checksum_sha256 | string | File hash |
| status | string | planned/acquired/quarantine/rejected |
| notes | string | Additional notes |

## Data Sources

### Owned SOP Data

| Field | Value |
|-------|-------|
| source_id | owned_sop_v1 |
| source_name | Fictional 3C Store SOP |
| license | CC0 (public domain) |
| status | acquired |
| notes | Self-generated fictional store policies |

### Public Datasets

| Dataset | License | Status | Notes |
|---------|---------|--------|-------|
| thu-coai/CrossWOZ | Apache-2.0 | planned | Chinese task-oriented dialogue |
| AmazonScience/massive | CC BY 4.0 | planned | Multilingual intent classification |
| bitext/Bitext-customer-support | CDLA-Sharing-1.0 | planned | English customer support |
