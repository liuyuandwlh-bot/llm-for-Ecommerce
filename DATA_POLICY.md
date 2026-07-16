# Data Governance Policy

## Overview

This document defines the data governance policy for the LLM Portfolio Platform project.

## Data Classification

### Level 1: Public/Open Data
- Synthetic data created by the project
- Data with clear open licenses (Apache 2.0, MIT, CC-BY, etc.)
- Government/public disclosures

### Level 2: Licensed Data
- Data with specific usage terms
- Requires attribution
- May have restrictions on redistribution

### Level 3: Proprietary/Restricted
- Data with unknown licensing
- Potentially contains PII
- Requires quarantine review before use

## Data Lifecycle

1. **Raw**: Immutable original data, never modified
2. **Interim**: Processed but not yet validated
3. **Processed**: Validated and ready for training/evaluation
4. **Quarantine**: Suspicious data pending review

## License Requirements

All data must have a documented license before entering the project. Unknown licenses require:
1. Quarantine storage
2. Explicit approval for use
3. Clear documentation of limitations

## PII Handling

Personal Identifiable Information (PII) includes:
- Phone numbers
- Email addresses
- Physical addresses
- ID numbers
- Order numbers
- Account names

PII must be masked with type placeholders (e.g., `<PHONE_1>`) before processing.

## Data Quality Gates

- [ ] Source documented
- [ ] License verified
- [ ] PII scan completed
- [ ] Deduplication performed
- [ ] Format standardization
- [ ] Quality check passed

## Retention & Deletion

- Raw data: Keep indefinitely (immutable)
- Interim data: Review quarterly
- Quarantine data: Review within 30 days
- Processed data: Keep with model versions
