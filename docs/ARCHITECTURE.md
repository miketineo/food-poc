# Hexis food platform v2 — target architecture

Interactive original (Excalidraw): https://link.excalidraw.com/readonly/zW85ffCvKDKQFeIyNFUZ?darkMode=true

The matcher in this repo is the "Matcher jobs" box: it runs async in the
ingestion/curation pipeline, auto-accepts only on overwhelming agreement,
and routes ambiguity to the human-in-the-loop review queue. The canonical
catalogue is versioned with provenance so meal logs can pin food versions.

```mermaid
flowchart TB
    subgraph clients [Clients]
        WEB[User web]
        MOB[User mobile<br/>SQLite offline queue]
    end

    subgraph ingestion [Ingestion / curation - async]
        SRC[3rd party sources<br/>heterogeneous schemas] --> ING[Ingestion svc]
        ING --> STAGE[Raw staging<br/>replayable]
        STAGE --> NORM[Normalize jobs]
        NORM --> MATCH[Matcher jobs]
        MATCH -- ambiguous --> RQ[Review queue<br/>human-in-loop]
    end

    WEB -- REST --> GW[API gateway]
    MOB -- "sync, idempotency keys" --> GW
    GW -- gRPC --> BE[Backend<br/>Food API / Log API]

    MATCH -- auto-accept --> CAT[Canonical catalogue<br/>versioned + provenance]
    RQ -- approved --> CAT

    CAT --> IDX[Indexer]
    IDX --> OS[OpenSearch - text]
    IDX --> KV[Barcode KV cache]

    BE -- search --> OS
    BE -- barcode get --> KV
    BE -- "log write, ack after persist" --> LOG[Meal log store<br/>pinned food versions]

    LOG -- log event --> BUS[Event bus]
    BUS --> FUEL[Fuelling engine<br/>recalc guidance]
    FUEL -. push updated guidance .-> LOG
```
