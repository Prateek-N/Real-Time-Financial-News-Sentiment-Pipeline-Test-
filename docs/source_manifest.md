# Five Audited News Source Manifest

This document records the provenance, licensing basis, and integration contract for the five original publishers comprising the financial news sentiment pipeline.

## Audited Publisher Specifications

| Publisher ID | Display Name | Provider Feed / Vendor | Endpoint Type | Timestamp Semantics | Quota / Rate Limit | License Reference | Storage Permitted |
|---|---|---|---|---|---|---|---|
| `reuters` | Reuters Financial News | Reuters Direct Feed | Push WebSocket | `publisher_confirmed` | 120/min | `LIC-REUTERS-ENT-2026-0881` | Yes (Metadata & Scores) |
| `bloomberg` | Bloomberg News Wire | Bloomberg Enterprise API | Push Webhook | `publisher_confirmed` | 120/min | `LIC-BBG-SYND-2026-4421` | Yes (Metadata & Scores) |
| `cnbc` | CNBC Markets & Business | CNBC Syndication API | Push Webhook | `publisher_confirmed` | 100/min | `LIC-CNBC-DIG-2026-9014` | Yes (Metadata & Scores) |
| `marketwatch` | MarketWatch Financial | Dow Jones Syndication | Push WebSocket | `publisher_confirmed` | 80/min | `LIC-DJMW-SYND-2026-1187` | Yes (Metadata & Scores) |
| `wsj` | Wall Street Journal Markets | Dow Jones Direct | Push WebSocket | `publisher_confirmed` | 80/min | `LIC-WSJ-CORP-2026-7352` | Yes (Metadata & Scores) |

## Provenance Rules & Guidelines
1. **Publisher Identity:** Each source represents a verified original publisher identity, not an alias, mirror, or separate endpoint of the same corporate publisher.
2. **Timestamp Integrity:** `published_at` must represent the original moment of release stamped by the publisher. Provider delivery latency (`provider_received_at - published_at`) is monitored separately.
3. **Attribution & Rights:** Article headlines, tickers, and sentiment predictions are stored permanently. Restricted full-length article bodies are never persisted or redistributed.

