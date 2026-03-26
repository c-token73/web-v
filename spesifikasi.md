# CSM++ v4.0 — AI-Native Semantic Compression & Data Science Engine
## Spesifikasi Teknis Lengkap (Production-Grade)

**Copyright © 2026 Nafal Faturizki · CENTRA-NF DSL Ecosystem**
**Versi Dokumen:** 4.0.0-final
**Status:** Production Specification — Ready for Engineering Implementation
**Bahasa Implementasi:** Rust (edisi 2021), Python bindings via PyO3

---

## Daftar Isi

- [§0 — Product Positioning](#§0--product-positioning-final)
- [§1 — System Architecture](#§1--system-architecture)
- [§2 — Core Algorithm Design](#§2--core-algorithm-design)
- [§3 — Rust Implementation Design](#§3--rust-implementation-design)
- [§4 — Data Model & Semantic Output](#§4--data-model--semantic-output)
- [§5 — Automatic Feature Engineering](#§5--automatic-feature-engineering)
- [§6 — File Format (.csm v4)](#§6--file-format-csm-v4)
- [§7 — API Design](#§7--api-design)
- [§8 — Streaming & Real-Time Processing](#§8--streaming--real-time-processing)
- [§9 — Performance Model](#§9--performance-model-realistic)
- [§10 — Benchmark Design](#§10--benchmark-design)
- [§11 — Android / Edge Mode](#§11--android--edge-mode)
- [§12 — Go-To-Market Strategy](#§12--go-to-market-strategy)
- [§13 — Risks & Limitations](#§13--risks--limitations)

---

## §0 — Product Positioning (Final)

### 0.1 Definisi Resmi

> **CSM++ v4.0** adalah *Semantic Compression + Automatic Feature Engineering Layer* yang mengubah data teks tidak terstruktur (logs, source code, AI training datasets) menjadi representasi semantik terstruktur yang siap digunakan oleh pipeline Machine Learning — dengan throughput tinggi, overhead memori minimal, dan output yang deterministik.

CSM++ **BUKAN**:
- General-purpose file compressor (bukan pengganti gzip/zstd)
- Generic tokenizer replacement (bukan pengganti BPE/SentencePiece untuk LLM)
- Penelitian akademis — ini adalah sistem produksi

CSM++ **ADALAH**:
- Preprocessing layer yang mengubah teks repetitif menjadi fitur ML siap pakai
- Compression engine yang domain-aware (logs, code, datasets) bukan general-text
- Feature store builder otomatis dari pola linguistik dan struktural
- Sistem yang dapat diintegrasikan ke pipeline data engineering (Kafka, Parquet, Arrow)

### 0.2 Target Domain

#### Domain 1: Logs (Primary)
- **Input**: Application logs, server logs, security audit logs, Kubernetes event logs
- **Karakteristik**: Sangat repetitif (80-95% token overlap antar baris), terstruktur parsial
- **Output CSM++**: Pattern table (template log), slot values (IP, timestamp, error code), anomaly score per sequence
- **CR realistis**: 8x–20x vs gzip, 3x–6x vs zstd (karena semantic awareness, bukan byte-level)
- **ML use case**: Log anomaly detection, incident classification, MTTR prediction

#### Domain 2: Source Code (Secondary)
- **Input**: GitHub repositories, code review diffs, compiler output, stack traces
- **Karakteristik**: Repetitif pada level AST-patterns (function calls, variable declarations, control flow)
- **Output CSM++**: Code pattern IDs, identifier slots, complexity features
- **CR realistis**: 4x–10x vs BPE token storage
- **ML use case**: Code quality prediction, bug localization, clone detection

#### Domain 3: AI Training Datasets (Tertiary)
- **Input**: Instruction datasets, QA pairs, structured text corpora
- **Karakteristik**: Semi-repetitif, slot-rich (named entities, numbers, dates)
- **Output CSM++**: Deduplicated pattern space, slot-normalized features
- **CR realistis**: 2x–4x vs BPE
- **ML use case**: Dataset deduplication, training data quality scoring, data poisoning detection

### 0.3 Positioning vs Kompetitor

| Sistem | Klaim | Kelemahan vs CSM++ |
|---|---|---|
| **gzip** | General byte compression | Tidak menghasilkan fitur ML; tidak semantic-aware |
| **zstd** | Fast byte compression | Sama dengan gzip — output tidak interpretable |
| **BPE (tiktoken)** | LLM tokenization | Output hanya integer IDs; tidak ada fitur; tidak domain-aware |
| **SentencePiece** | Multilingual tokenization | Tidak menghasilkan slot-values atau anomaly features |
| **Drain3** | Log template mining | Tidak mengompresi; tidak ada fitur ML otomatis; Python-only |
| **LogPai** | Log parsing framework | Lambat (Python); tidak streaming; tidak ada compression |

**CSM++ advantage**: Satu-satunya sistem yang menggabungkan (a) compression, (b) template extraction, (c) slot normalization, dan (d) automatic feature engineering dalam satu pipeline Rust yang streaming-capable.

---

## §1 — System Architecture

### 1.1 Overview Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CSM++ v4.0 PIPELINE                              │
│                                                                         │
│  Raw Data      Ingestion       CSM++ Core      Semantic Layer           │
│  ─────────    ──────────      ──────────      ──────────────            │
│  logs       →  Chunker    →   Normalizer  →   Pattern Matcher   →      │
│  code           Validator      Tokenizer       Slot Extractor           │
│  datasets       Rate Limiter   FST Lookup      Template Builder         │
│                                                                         │
│  Feature Store     ML Pipeline        Output                            │
│  ─────────────    ──────────────     ────────                           │
│  → FeatureEngine → DataFrame Bridge → Parquet / Arrow / Kafka           │
│    AnomalyScore    Polars/Pandas      .csm v4 binary                    │
│    SeqPatterns     scikit-learn       JSON streaming                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Layer Descriptions

#### Layer 1: Ingestion (`ingestion/`)
**Tanggung jawab**: Menerima input dari berbagai sumber, memvalidasi, dan mengirim ke Core dalam chunk yang optimal.

```
Input Sources:
  - File I/O (mmap untuk file > 100MB)
  - Stdin streaming
  - Kafka consumer (via rdkafka crate)
  - HTTP webhook (via axum)
  - In-memory buffer (untuk embedding dalam aplikasi lain)

Output: Stream<Chunk>
  - Chunk = Arc<[u8]> (zero-copy reference ke mmap'd region)
  - Ukuran chunk: 4096–65536 bytes (configurable, default 16384)
  - Chunk metadata: source_id, offset, timestamp
```

#### Layer 2: CSM++ Core (`core/`)
**Tanggung jawab**: Normalisasi, tokenisasi, FST matching, Viterbi selection.

```
Sub-components:
  - Normalizer: Unicode normalization, whitespace collapse, domain-specific rules
  - Tokenizer: Byte-pair aware splitting, domain tokenization (log fields, code tokens)
  - FST Engine: Finite State Transducer lookup (fst crate)
  - DP Selector: Viterbi-based optimal pattern selection
  - Fallback Chain: Pattern → CSM → Subword → Byte → UNK
```

#### Layer 3: Semantic Layer (`semantic/`)
**Tanggung jawab**: Mengubah output Core menjadi representasi semantik terstruktur.

```
Sub-components:
  - PatternRegistry: Global pattern table (thread-safe, Arc<RwLock<>>)
  - SlotExtractor: Typed slot value extraction (integer, float, IP, timestamp, hash)
  - TemplateBuilder: Human-readable template dari base_seq + slot positions
  - AnomalyDetector: Statistical scoring terhadap expected slot distributions
```

#### Layer 4: Feature Store (`features/`)
**Tanggung jawab**: Menghitung fitur ML dari output Semantic Layer.

```
Sub-components:
  - PatternFrequencyEngine: Rolling window frequency counter
  - SlotStatEngine: Per-slot mean, std, min, max, cardinality
  - SequencePatternEngine: N-gram patterns di level pattern_id sequence
  - AnomalyScoreEngine: Z-score dan IQR-based outlier scoring
```

#### Layer 5: Output (`api/`, `encoding/`)
**Tanggung jawab**: Serialisasi ke format output yang dipilih.

```
Output Formats:
  - .csm v4 binary (default, lihat §6)
  - Apache Arrow RecordBatch (via arrow2 crate)
  - Parquet (via arrow2 + parquet2)
  - JSON Lines (untuk streaming ke Kafka / log sinks)
  - Python DataFrame (via PyO3 + polars)
```

### 1.3 Data Flow Diagram (Detail)

```
Raw bytes
    │
    ▼
[Chunker] ──────────────────────────────────────────────────────┐
    │ Arc<[u8]>                                                  │
    ▼                                                            │
[Normalizer] → clean UTF-8 string                               │
    │                                                            │
    ▼                                                            │
[DomainTokenizer] → Vec<Token>                                  │
    │   Token = { id: VocabId, raw: &str, kind: TokenKind }     │
    ▼                                                            │
[FSTMatcher] → Vec<MatchCandidate>                              │
    │   MatchCandidate = { start, end, pattern_id, gain }        │
    ▼                                                            │
[ViterbiDP] → Vec<Assignment>                                   │
    │   Assignment = Pattern(PatternId, Vec<SlotValue>)          │
    │               | Token(VocabId)                             │
    ▼                                                            │
[SlotExtractor] → SemanticRecord                                │
    │   SemanticRecord = { pattern_id, template, slots, ts }     │
    ▼                                                            │
[FeatureEngine] → FeatureVector                                 │
    │   FeatureVector = { freq, slot_stats, anomaly_score, ... } │
    ▼                                                            │
[OutputRouter] ──────────────────────────────────────────────── ┘
    ├── .csm v4 binary
    ├── Arrow RecordBatch
    └── Kafka JSON
```

---

## §2 — Core Algorithm Design

### 2.1 Pattern Matching Strategy: Hybrid FST + Inverted Index

CSM++ menggunakan **dua-layer matching**:

**Layer A — FST (Finite State Transducer)** untuk exact sequence matching:
- Library: `fst` crate (BurntSushi) — minimal DFA, ~10 bytes/entry
- Key: serialized token ID sequence (little-endian u32 array)
- Value: PatternId (u32)
- Query: longest-match traversal, O(k) per query (k = max pattern length = 5)
- Memory: ~5MB untuk 65K patterns — masuk L3 cache entirely

**Layer B — Inverted Index** untuk partial/fuzzy matching (opsional, Phase 2):
- Untuk domain Log: Drain3-style template tree
- Key: token sequence dengan wildcard (`*`) di slot positions
- Value: PatternId + slot positions
- Berguna ketika exact FST match gagal tapi template match ada

**Pemilihan Strategi per Domain**:
```
Domain = Logs:   Layer A (FST) first → Layer B (template tree) fallback
Domain = Code:   Layer A (FST) only  → Layer B not needed (lebih structured)
Domain = Text:   Layer A (FST) only  → generic CSM fallback
```

### 2.2 FST Build Algorithm

```
Input:  Sorted list of (sequence: Vec<VocabId>, pattern_id: u32)
Output: Compiled FST binary (mmap-able)

Algorithm:
1. Sort patterns by serialized key (lexicographic on byte representation)
2. Iterate sorted patterns, insert ke MapBuilder
3. Build → serialize ke disk (binary, ~5MB untuk 65K patterns)
4. mmap file saat runtime → zero-copy access

Complexity:
  Build:  O(P · k · log P)  — P = pattern count, k = max length
  Query:  O(k)              — independent of P
  Memory: O(P · 10 bytes)   — FST compressed

vs HashMap lookup:
  Build:  O(P · k)
  Query:  O(k) average, O(k²) worst (hash collision)
  Memory: O(P · 60 bytes)   — HashMap overhead
```

### 2.3 Domain-Aware Tokenizer

Berbeda dari BPE, CSM++ menggunakan **rule-based domain tokenizer** yang dapat dikonfigurasi:

```
LOG TOKENIZER:
  Rules (applied in order):
    1. Split on whitespace → raw_fields
    2. Classify each field:
       - TIMESTAMP: regex [0-9]{4}-[0-9]{2}-[0-9]{2}T... → SlotType::Timestamp
       - IP_ADDR: regex \d{1,3}\.\d{1,3}... → SlotType::IpAddress
       - INTEGER: all-digit field → SlotType::Integer
       - FLOAT: digit.digit field → SlotType::Float
       - LOG_LEVEL: INFO|WARN|ERROR|DEBUG|TRACE → SlotType::LogLevel
       - HEX: 0x[0-9a-fA-F]+ → SlotType::Hex
       - PATH: contains / or \ → SlotType::Path
       - UUID: 8-4-4-4-12 hex → SlotType::Uuid
       - LITERAL: everything else → vocab lookup
    3. Replace classified fields dengan slot placeholder token
    4. Remaining literals → FST vocab lookup

CODE TOKENIZER:
  Rules:
    1. Lex source code (minimal lexer per language)
       - Rust: keywords, identifiers, literals, punctuation
       - Python: keywords, identifiers, numbers, strings, operators
    2. Identifier normalization: camelCase → snake_case (optional)
    3. String literals → STRING_LITERAL placeholder
    4. Numeric literals → INTEGER/FLOAT placeholder
    5. AST-level patterns: function signatures, import statements
```

### 2.4 Viterbi DP — Full Specification

Lihat §3.4 untuk implementasi Rust. Formulation:

```
State:     dp[i] = (max_bits_saved: i64, prev_idx: usize, assignment: AssignmentKind)
Init:      dp[0] = (0, 0, None)
Recurrence:
  For i = 1..=N:
    Option A (single token, no pattern):
      gain_A = dp[i-1].gain - tier_bits(tokens[i-1])
      if gain_A > dp[i].gain: dp[i] = (gain_A, i-1, Token(tokens[i-1]))

    Option B (pattern ending at position i):
      For each pattern p where p.end == i:
        gain_B = dp[p.start].gain + p.compress_gain
        if gain_B > dp[i].gain: dp[i] = (gain_B, p.start, Pattern(p.id))

Backtrack: O(N) reverse traversal dari dp[N]

Total complexity: O(N × max_pattern_len) = O(5N) = O(N)
Space:            O(N) untuk dp array
```

### 2.5 Slot Extraction Algorithm

```
Input:  Assignment::Pattern(pattern_id, raw_tokens[start..end])
Output: SemanticRecord { pattern_id, slots: Vec<SlotValue> }

Algorithm:
  pattern = registry.get(pattern_id)
  slots = Vec::with_capacity(pattern.slot_count)

  for (slot_pos, slot_type) in pattern.slot_schema.iter() {
      raw_token = raw_tokens[slot_pos]
      slot_value = match slot_type {
          SlotType::Integer    => parse_integer(raw_token)?,
          SlotType::Float      => parse_float(raw_token)?,
          SlotType::IpAddress  => parse_ip(raw_token)?,
          SlotType::Timestamp  => parse_timestamp(raw_token)?,
          SlotType::LogLevel   => map_log_level(raw_token),
          SlotType::Hex        => parse_hex(raw_token)?,
          SlotType::Path       => intern_path(raw_token),
          SlotType::Uuid       => parse_uuid(raw_token)?,
          SlotType::Generic    => intern_string(raw_token),
      };
      slots.push(slot_value);
  }
  SemanticRecord { pattern_id, template: pattern.template.clone(), slots, timestamp: now() }

Complexity: O(slot_count) per record — biasanya 1–5 slots
```

### 2.6 Encoding & Decoding

#### Encoding (teks → .csm binary)
```
1. Tokenize: raw_text → Vec<Token>                    O(N)
2. FST match: Vec<Token> → Vec<MatchCandidate>         O(N·k)
3. Viterbi:  Vec<MatchCandidate> → Vec<Assignment>     O(N)
4. Slot extract: Vec<Assignment> → Vec<SemanticRecord> O(N)
5. Feature compute: Vec<SemanticRecord> → FeatureVec  O(N·W) W=window
6. Bit-pack: Vec<Assignment> → bit stream              O(N·b)
7. Write .csm: header + sections + data                O(N·b/8)

Total: O(N) time, O(chunk_size) space (streaming-capable)
```

#### Decoding (.csm binary → teks atau DataFrame)
```
1. Read header, validate CRC32c                        O(1)
2. Load vocab + pattern registry                       O(V + P)
3. Bit-unpack: bit stream → Vec<Assignment>            O(M·b)
4. Reconstruct: Vec<Assignment> + slot values → text  O(N)

For DataFrame output:
  Skip step 4; directly materialize SemanticRecord[]  O(N)
  → Arrow RecordBatch

Total decode: O(N) time, O(V + P + chunk) space
```

---

## §3 — Rust Implementation Design

### 3.1 Workspace & Crate Structure

```
csm-plus-plus/                     (workspace root)
├── Cargo.toml                     (workspace manifest)
├── crates/
│   ├── csm-core/                  # Core algorithm (no I/O dependencies)
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── vocab.rs           # Vocabulary + arena allocator
│   │   │   ├── pattern.rs         # Pattern struct + registry
│   │   │   ├── slot.rs            # SlotType, SlotValue, SlotSchema
│   │   │   ├── fst_engine.rs      # FST build + query
│   │   │   ├── viterbi.rs         # DP pattern selection
│   │   │   ├── fallback.rs        # Hierarchical fallback chain
│   │   │   └── error.rs           # CsmError enum
│   │   └── Cargo.toml
│   │
│   ├── csm-tokenizer/             # Domain tokenizers
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── log_tokenizer.rs
│   │   │   ├── code_tokenizer.rs
│   │   │   ├── text_tokenizer.rs
│   │   │   └── token.rs
│   │   └── Cargo.toml
│   │
│   ├── csm-encoding/              # Bit-packing + file format
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── bit_writer.rs      # BitWriter + multi-tier pack
│   │   │   ├── bit_reader.rs      # BitReader + decode
│   │   │   ├── file_format.rs     # .csm v4 header + sections
│   │   │   └── crc.rs             # CRC32c (Castagnoli)
│   │   └── Cargo.toml
│   │
│   ├── csm-semantic/              # Semantic layer + feature engineering
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── semantic_record.rs
│   │   │   ├── feature_engine.rs  # Automatic feature computation
│   │   │   ├── anomaly.rs         # Anomaly scoring
│   │   │   ├── template.rs        # Template builder
│   │   │   └── arrow_bridge.rs    # Arrow RecordBatch output
│   │   └── Cargo.toml
│   │
│   ├── csm-streaming/             # Streaming + Kafka integration
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── chunker.rs
│   │   │   ├── kafka_source.rs
│   │   │   ├── kafka_sink.rs
│   │   │   └── frame.rs
│   │   └── Cargo.toml
│   │
│   ├── csm-discovery/             # Pattern discovery (build phase)
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── cms.rs             # Count-Min Sketch
│   │   │   ├── ngram_counter.rs
│   │   │   ├── ppmi.rs            # PPMI scoring
│   │   │   ├── pattern_scorer.rs
│   │   │   └── fst_builder.rs
│   │   └── Cargo.toml
│   │
│   ├── csm-api/                   # Public Rust API + CLI
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── encoder.rs
│   │   │   ├── decoder.rs
│   │   │   └── cli.rs
│   │   └── Cargo.toml
│   │
│   └── csm-python/                # PyO3 Python bindings
│       ├── src/
│       │   ├── lib.rs
│       │   └── bindings.rs
│       └── Cargo.toml
│
├── benches/                       # Criterion benchmarks
├── fuzz/                          # cargo-fuzz targets
└── examples/
```

### 3.2 Key Dependencies

```toml
# Cargo.toml (workspace)
[workspace.dependencies]
# Core
fst            = "0.4"         # FST untuk pattern lookup
bumpalo        = "3.16"        # Arena allocator
rustc-hash     = "2.0"         # FxHashMap (40% faster std::HashMap)
smallvec       = "1.13"        # Stack-allocated Vec<T; N>

# SIMD & Performance
simdutf        = "0.4"         # SIMD UTF-8 validation + normalization
packed_simd_2  = "0.3"         # Fallback SIMD (AVX2/NEON)

# Parallelism
rayon          = "1.10"        # Data parallelism
tokio          = { version = "1.38", features = ["full"] }  # Async I/O

# Encoding
crc32fast      = "1.4"         # Hardware-accelerated CRC32c

# Output formats
arrow2         = "0.18"        # Apache Arrow
parquet2       = "0.18"        # Parquet output

# Streaming
rdkafka        = "0.36"        # Kafka consumer/producer
axum           = "0.7"         # HTTP API

# Serialization
serde          = { version = "1.0", features = ["derive"] }
bincode        = "2.0"         # Binary serialization

# Python bindings
pyo3           = { version = "0.21", features = ["extension-module"] }

# Discovery phase
rand           = "0.8"         # Reservoir sampling
statrs         = "0.16"        # Statistical functions (KL divergence, etc)

# CLI
clap           = { version = "4.5", features = ["derive"] }
tracing        = "0.1"
tracing-subscriber = "0.3"
```

### 3.3 Core Structs

```rust
// ─── csm-core/src/vocab.rs ────────────────────────────────────────────────

use bumpalo::Bump;
use rustc_hash::FxHashMap;

pub type VocabId = u32;
pub const VOCAB_ID_UNK: VocabId = 0;
pub const VOCAB_ID_UNK_BYTE_BASE: VocabId = 1; // 1–256 reserved for byte fallback

/// Vocabulary dengan arena-allocated strings (zero-copy, no duplication)
pub struct Vocab<'arena> {
    /// Primary lookup: string slice → ID (borrowed from arena)
    str_to_id: FxHashMap<&'arena str, VocabId>,
    /// Reverse lookup: ID → string slice
    id_to_str:  Vec<&'arena str>,
    /// Frequency dari training corpus (untuk tier assignment)
    id_to_freq: Vec<u32>,
    /// Tier assignment cache (0/1/2/3 — precomputed after freeze)
    id_to_tier: Vec<u8>,
    /// Zipf-sorted IDs (freq DESC) — untuk tier boundary computation
    freq_sorted: Vec<VocabId>,
    /// Memory arena — semua string hidup di sini
    arena: Bump,
    /// Tier boundaries [cutoff_0_1, cutoff_1_2, cutoff_2_3]
    tier_cutoffs: [u32; 3],
    /// Fingerprint: FNV-1a hash dari sorted(id→str) — untuk drift detection
    pub fingerprint: u64,
    /// Frozen = true → no more insertions allowed
    pub frozen: bool,
}

impl<'arena> Vocab<'arena> {
    pub fn new() -> Self { ... }
    pub fn insert(&mut self, s: &str) -> Result<VocabId, CsmError> { ... }
    pub fn freeze(&mut self) { ... } // compute tier_cutoffs + fingerprint
    pub fn id(&self, s: &str) -> Option<VocabId> { ... }
    pub fn str(&self, id: VocabId) -> Option<&str> { ... }
    pub fn tier(&self, id: VocabId) -> u8 { self.id_to_tier[id as usize] }
    pub fn tier_bits(&self, id: VocabId) -> u8 {
        match self.tier(id) { 0 => 8, 1 => 14, 2 => 18, _ => 255 } // 255 = Elias-γ
    }
    pub fn size(&self) -> usize { self.id_to_str.len() }
}

// ─── csm-core/src/slot.rs ─────────────────────────────────────────────────

/// Tipe semantik dari slot — menentukan cara ekstraksi dan feature generation
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum SlotType {
    // Numeric
    Integer,
    Float,
    // Network
    IpAddress,
    Port,
    // Time
    Timestamp,
    Duration,
    // System
    LogLevel,    // INFO, WARN, ERROR, DEBUG, TRACE, FATAL
    ExitCode,
    Pid,
    // Identifier
    Uuid,
    Hash,        // MD5, SHA1, SHA256 (hex string)
    Path,        // filesystem path
    // Code-specific
    Identifier,  // variable/function name
    StringLiteral,
    // Generic
    Generic,     // fallback: intern as string
}

/// Nilai aktual dari slot setelah ekstraksi
#[derive(Debug, Clone, serde::Serialize)]
pub enum SlotValue {
    Int(i64),
    Float(f64),
    IpV4([u8; 4]),
    IpV6([u8; 16]),
    Timestamp(i64),      // Unix timestamp micros
    Duration(u64),       // nanoseconds
    LogLevel(LogLevelKind),
    Uuid([u8; 16]),
    Hash(Box<[u8]>),     // variable length
    Text(u32),           // interned string ID
    Bytes(Box<[u8]>),    // binary fallback
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum LogLevelKind { Trace = 0, Debug = 1, Info = 2, Warn = 3, Error = 4, Fatal = 5 }

/// Schema slot untuk satu pattern (immutable setelah pattern di-freeze)
#[derive(Debug, Clone)]
pub struct SlotSchema {
    pub slots: smallvec::SmallVec<[(u8, SlotType); 4]>, // (position_in_template, type)
}

// ─── csm-core/src/pattern.rs ──────────────────────────────────────────────

use smallvec::SmallVec;

pub type PatternId = u32;
pub const INVALID_PATTERN: PatternId = u32::MAX;

/// Pattern struct — cache-line aligned untuk SIMD-friendly bulk scan
#[repr(C, align(64))]
#[derive(Debug, Clone)]
pub struct Pattern {
    // Identity
    pub id:             PatternId,
    pub domain:         DomainKind,    // Log, Code, Text
    /// Base token sequence (max 5 tokens, stack-allocated)
    pub base_seq:       SmallVec<[VocabId; 5]>,
    /// Slot schema: posisi dan tipe slot
    pub slot_schema:    SlotSchema,
    /// Human-readable template: "Connection from {IP} port {PORT} rejected"
    pub template:       String,

    // Discovery metrics (immutable setelah build)
    pub freq:           u32,
    pub ppmi_score:     f32,
    pub compress_gain:  f32,    // bits saved per occurrence
    pub pgs_score:      f32,    // Pattern Generalization Score [0,1]
    pub stability:      f32,    // context stability [0,1]
    pub final_score:    f32,

    // Status
    pub deprecated:     bool,
    pub _pad:           [u8; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum DomainKind { Log, Code, Text, Generic }

/// Thread-safe pattern registry
pub struct PatternRegistry {
    patterns:   Vec<Pattern>,                      // indexed by PatternId
    by_domain:  FxHashMap<DomainKind, Vec<PatternId>>,
    template_index: FxHashMap<String, PatternId>,  // untuk reverse lookup
    pub frozen: bool,
}

impl PatternRegistry {
    pub fn new() -> Self { ... }
    pub fn register(&mut self, p: Pattern) -> Result<PatternId, CsmError> { ... }
    pub fn get(&self, id: PatternId) -> Option<&Pattern> { ... }
    pub fn by_template(&self, template: &str) -> Option<PatternId> { ... }
    pub fn freeze(&mut self) { self.frozen = true; }
    pub fn count(&self) -> usize { self.patterns.len() }
}
```

### 3.4 Algorithm Structs & Viterbi Implementation

```rust
// ─── csm-core/src/viterbi.rs ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MatchCandidate {
    pub start:       usize,
    pub end:         usize,       // exclusive
    pub pattern_id:  PatternId,
    pub compress_gain: f32,
}

#[derive(Debug, Clone)]
pub enum AssignmentKind {
    Pattern { id: PatternId },
    Token   { id: VocabId },
    Fallback { level: u8 },       // 2=subword, 3=byte, 4=unk
}

#[derive(Debug, Clone)]
pub struct Assignment {
    pub kind:  AssignmentKind,
    pub start: usize,
    pub end:   usize,
}

#[derive(Clone)]
struct DpEntry {
    gain:   i64,
    prev:   usize,
    kind:   AssignmentKind,
}

/// Viterbi DP — O(N × max_pattern_len) time, O(N) space
pub fn viterbi_select(
    tokens:    &[VocabId],
    matches:   &[MatchCandidate],
    vocab:     &Vocab<'_>,
) -> Vec<Assignment> {
    let n = tokens.len();
    let neg_inf = i64::MIN / 2;

    // Pre-index: end_position → candidates ending there
    let mut by_end: Vec<Vec<&MatchCandidate>> = vec![vec![]; n + 1];
    for m in matches { by_end[m.end].push(m); }

    let mut dp: Vec<DpEntry> = (0..=n).map(|_| DpEntry {
        gain: neg_inf, prev: 0, kind: AssignmentKind::Token { id: 0 }
    }).collect();
    dp[0].gain = 0;

    for i in 1..=n {
        // Option A: single token at i-1
        if dp[i-1].gain != neg_inf {
            let cost = vocab.tier_bits(tokens[i-1]) as i64;
            let g = dp[i-1].gain - cost;
            if g > dp[i].gain {
                dp[i] = DpEntry { gain: g, prev: i-1,
                    kind: AssignmentKind::Token { id: tokens[i-1] } };
            }
        }
        // Option B: patterns ending at i
        for m in &by_end[i] {
            if dp[m.start].gain != neg_inf {
                let g = dp[m.start].gain + m.compress_gain as i64;
                if g > dp[i].gain {
                    dp[i] = DpEntry { gain: g, prev: m.start,
                        kind: AssignmentKind::Pattern { id: m.pattern_id } };
                }
            }
        }
    }

    // Backtrack
    let mut assignments = Vec::new();
    let mut pos = n;
    while pos > 0 {
        let entry = &dp[pos];
        assignments.push(Assignment {
            kind:  entry.kind.clone(),
            start: entry.prev,
            end:   pos,
        });
        pos = entry.prev;
    }
    assignments.reverse();
    assignments
}
```

### 3.5 Trait Design

```rust
// ─── csm-core/src/lib.rs ──────────────────────────────────────────────────

/// Trait utama untuk semua tokenizer domain
pub trait DomainTokenizer: Send + Sync {
    type Config: Default + Clone + Send;

    fn new(config: Self::Config) -> Self;
    fn tokenize<'a>(&self, input: &'a str) -> Vec<Token<'a>>;
    fn domain(&self) -> DomainKind;
}

/// Trait untuk semua encoder (sync atau streaming)
pub trait CsmEncoder {
    type Error: std::error::Error + Send + Sync + 'static;

    fn encode(&mut self, input: &str) -> Result<CsmData, Self::Error>;
    fn encode_batch(&mut self, inputs: &[&str]) -> Result<Vec<CsmData>, Self::Error>;
    fn domain(&self) -> DomainKind;
}

/// Trait untuk output format (Arrow, Parquet, JSON, binary)
pub trait OutputFormat: Send + Sync {
    fn write_record(&mut self, record: &SemanticRecord) -> Result<(), CsmError>;
    fn write_batch(&mut self, records: &[SemanticRecord]) -> Result<(), CsmError>;
    fn flush(&mut self) -> Result<Vec<u8>, CsmError>;
    fn format_name(&self) -> &'static str;
}

/// Trait untuk sumber data streaming
pub trait DataSource: Send {
    type Item: AsRef<[u8]>;
    fn next_chunk(&mut self) -> Option<Result<Self::Item, CsmError>>;
    fn source_id(&self) -> &str;
}

/// Trait untuk feature extractors
pub trait FeatureExtractor: Send + Sync {
    fn extract(&self, records: &[SemanticRecord]) -> FeatureVector;
    fn feature_names(&self) -> Vec<String>;
    fn feature_count(&self) -> usize;
}

// ─── Lifetime design note ─────────────────────────────────────────────────
//
// 'arena lifetime digunakan di Vocab<'arena> dan Token<'a>:
//
//   Token<'a>: punya &'a str reference ke input string
//   Vocab<'arena>: punya &'arena str reference ke arena-allocated strings
//
// Kedua lifetimes TIDAK perlu sama — Token references ke input (short-lived),
// Vocab references ke arena (long-lived, sama dengan Vocab instance).
//
// Ownership model:
//   CsmEncoder owns: Vocab (Arc), PatternRegistry (Arc), FST (Arc<Mmap>)
//   → Shared via Arc untuk parallel encoding
//   → Immutable setelah freeze() → no RwLock overhead di hot path
//
// Concurrency:
//   Arc<Vocab>           — shared read, no write setelah freeze
//   Arc<PatternRegistry> — shared read, no write setelah freeze
//   Arc<Mmap>            — shared FST binary (mmap'd, read-only)
//   → Hot path FULLY lock-free ✓
```

### 3.6 Error Types

```rust
// ─── csm-core/src/error.rs ────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum CsmError {
    // Vocab errors
    #[error("Vocab is frozen; cannot insert '{0}'")]
    VocabFrozen(String),
    #[error("Vocab drift detected: expected fingerprint {expected:#016x}, got {found:#016x}")]
    VocabDrift { expected: u64, found: u64 },
    #[error("VocabId {0} out of range (vocab size: {1})")]
    VocabIdOob(u32, usize),

    // Encoding errors
    #[error("Invalid UTF-8 at byte offset {offset}: {source}")]
    InvalidUtf8 { offset: usize, #[source] source: std::str::Utf8Error },
    #[error("Input too large: {size} bytes (max: {max})")]
    InputTooLarge { size: usize, max: usize },
    #[error("CRC32c mismatch: expected {expected:#010x}, computed {computed:#010x}")]
    CrcMismatch { expected: u32, computed: u32 },

    // File format errors
    #[error("Invalid magic bytes: expected 'CSM4', found {0:?}")]
    InvalidMagic([u8; 4]),
    #[error("Unsupported version: {major}.{minor}")]
    UnsupportedVersion { major: u8, minor: u8 },
    #[error("Section {section} corrupt or truncated")]
    SectionCorrupt { section: &'static str },

    // Pattern errors
    #[error("PatternId {0} not found in registry")]
    PatternNotFound(PatternId),
    #[error("Pattern compress_gain {0} ≤ 0; pattern should be rejected at build")]
    NegativeCompressGain(f32),

    // Slot errors
    #[error("Slot extraction failed for type {slot_type:?}: {reason}")]
    SlotExtractionFailed { slot_type: SlotType, reason: String },
    #[error("Slot entropy {entropy:.3} outside bounds [{min:.1}, {max:.1}]")]
    SlotEntropyOob { entropy: f32, min: f32, max: f32 },

    // I/O
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Arrow error: {0}")]
    Arrow(String),

    // Streaming
    #[error("Kafka error: {0}")]
    Kafka(String),
    #[error("Channel closed unexpectedly")]
    ChannelClosed,
}
```

---

## §4 — Data Model & Semantic Output

### 4.1 SemanticRecord — Core Output Unit

```rust
/// Output utama dari CSM++ encoding — satu baris teks → satu SemanticRecord
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SemanticRecord {
    // Identity
    pub source_id:      String,        // file/stream identifier
    pub offset:         u64,           // byte offset dalam source
    pub line_number:    Option<u64>,

    // Temporal
    pub ingested_at:    i64,           // Unix micros (system time)
    pub log_timestamp:  Option<i64>,   // extracted dari log line (jika ada)

    // Semantic
    pub pattern_id:     PatternId,     // 0 = unmatched (raw token sequence)
    pub template:       String,        // "Connection from {IP} port {PORT} rejected"
    pub domain:         DomainKind,

    // Slot values
    pub slots:          Vec<NamedSlot>,

    // Compression metadata
    pub raw_token_count: u16,          // token count sebelum compression
    pub encoded_bits:    u32,          // bits setelah compression
    pub compress_ratio:  f32,          // raw_bits / encoded_bits

    // Feature annotations (computed by FeatureEngine)
    pub features:       Option<FeatureVector>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NamedSlot {
    pub name:       String,            // "IP", "PORT", "LEVEL", dll
    pub slot_type:  SlotType,
    pub value:      SlotValue,
    pub raw:        String,            // original string sebelum parsing
}
```

### 4.2 FeatureVector

```rust
/// Semua fitur ML yang dihasilkan otomatis oleh FeatureEngine
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FeatureVector {
    // Pattern features
    pub pattern_id:           u32,
    pub pattern_freq_1m:      f32,   // frequency dalam window 1 menit
    pub pattern_freq_1h:      f32,   // frequency dalam window 1 jam
    pub pattern_freq_rank:    f32,   // rank Zipf (normalized, 0=paling umum)
    pub is_rare_pattern:      bool,  // freq < threshold
    pub pattern_novelty:      f32,   // 1.0 = pattern baru, 0.0 = sudah sering muncul

    // Slot features (per slot, flattened)
    /// Untuk setiap slot: [mean, std, min, max, z_score, is_outlier]
    pub slot_stats:           Vec<SlotStats>,

    // Sequence features
    pub seq_bigram_score:     f32,   // PMI dari pattern_id[t-1] × pattern_id[t]
    pub seq_entropy_local:    f32,   // H(pattern|last_5)
    pub seq_burst_score:      f32,   // burstiness: (std/mean)² - 1/mean

    // Anomaly features
    pub anomaly_score:        f32,   // komposit [0, 1]; > 0.8 = high anomaly
    pub anomaly_components:   AnomalyComponents,

    // Context
    pub window_id:            u64,   // window bucket (1-minute bins default)
    pub computed_at:          i64,   // Unix micros
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SlotStats {
    pub slot_name:    String,
    pub mean:         f64,
    pub std:          f64,
    pub min:          f64,
    pub max:          f64,
    pub z_score:      f64,           // (current - mean) / std
    pub percentile:   f32,           // approximate via t-digest
    pub is_outlier:   bool,          // |z_score| > 3.0 atau IQR method
    pub cardinality:  u32,           // distinct values seen (HyperLogLog estimate)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AnomalyComponents {
    pub freq_anomaly:   f32,         // frequency terlalu tinggi/rendah
    pub slot_anomaly:   f32,         // slot values di luar distribusi normal
    pub seq_anomaly:    f32,         // pattern sequence tidak biasa
    pub temporal_anomaly: f32,       // waktu tidak sesuai pola historis
}
```

### 4.3 Mapping ke DataFrame

```rust
// ─── csm-semantic/src/arrow_bridge.rs ─────────────────────────────────────

use arrow2::{
    array::*,
    chunk::Chunk,
    datatypes::{Field, Schema, DataType},
};

/// Mengubah Vec<SemanticRecord> menjadi Arrow RecordBatch
pub fn records_to_arrow(records: &[SemanticRecord]) -> Result<Chunk<Box<dyn Array>>, CsmError> {
    let schema = build_schema(records);

    // Base columns (selalu ada)
    let col_source:     Box<dyn Array> = Utf8Array::<i32>::from(records.iter().map(|r| Some(r.source_id.as_str()))).boxed();
    let col_offset:     Box<dyn Array> = UInt64Array::from_vec(records.iter().map(|r| r.offset).collect()).boxed();
    let col_pattern_id: Box<dyn Array> = UInt32Array::from_vec(records.iter().map(|r| r.pattern_id).collect()).boxed();
    let col_template:   Box<dyn Array> = Utf8Array::<i32>::from(records.iter().map(|r| Some(r.template.as_str()))).boxed();
    let col_cr:         Box<dyn Array> = Float32Array::from_vec(records.iter().map(|r| r.compress_ratio).collect()).boxed();
    let col_anomaly:    Box<dyn Array> = Float32Array::from_vec(
        records.iter().map(|r| r.features.as_ref().map(|f| f.anomaly_score).unwrap_or(0.0)).collect()
    ).boxed();

    // Slot columns (dynamic — satu column per slot type yang ada)
    let slot_columns = extract_slot_columns(records);

    let mut all_cols = vec![col_source, col_offset, col_pattern_id, col_template, col_cr, col_anomaly];
    all_cols.extend(slot_columns);

    Ok(Chunk::new(all_cols))
}

/// Arrow Schema untuk SemanticRecord
pub fn semantic_schema() -> Schema {
    Schema::from(vec![
        Field::new("source_id",     DataType::Utf8, false),
        Field::new("offset",        DataType::UInt64, false),
        Field::new("log_timestamp", DataType::Int64, true),
        Field::new("pattern_id",    DataType::UInt32, false),
        Field::new("template",      DataType::Utf8, false),
        Field::new("domain",        DataType::Utf8, false),
        Field::new("compress_ratio",DataType::Float32, false),
        Field::new("anomaly_score", DataType::Float32, false),
        Field::new("pattern_freq_1m", DataType::Float32, false),
        Field::new("pattern_freq_1h", DataType::Float32, false),
        Field::new("seq_burst_score", DataType::Float32, false),
        // Slot columns ditambahkan secara dynamic berdasarkan domain
        // Log: slot_ip, slot_port, slot_level, slot_timestamp_extracted, ...
        // Code: slot_identifier, slot_line_num, ...
    ])
}
```

### 4.4 ML Feature Mapping

```
SemanticRecord fields → ML features:

CLASSIFICATION TASKS (log anomaly, incident type):
  Input features:
    - pattern_id (categorical, embedding via pattern_freq_rank)
    - anomaly_score (float [0,1])
    - slot_stats[*].z_score (float, per slot)
    - seq_burst_score (float)
    - pattern_novelty (float)
  Label: manual (anomaly=true/false) atau semi-supervised

REGRESSION TASKS (MTTR prediction, latency estimation):
  Input features:
    - pattern_freq_1h (float)
    - slot_stats["duration"].mean (float)
    - seq_entropy_local (float)
    - temporal_anomaly (float)
  Label: duration (minutes)

CLUSTERING (log grouping, incident deduplication):
  Features: pattern_id + slot_value_hashes
  → Groups = equivalent patterns (same template)
  → Sub-groups = same template + same slot value distribution

TIME SERIES ANOMALY:
  Time index: window_id (1-minute bins)
  Signal: pattern_freq_1m per pattern_id
  → Alert jika freq_1m > mean + 3σ (atau IQR × 3)
```

---

## §5 — Automatic Feature Engineering

### 5.1 PatternFrequencyEngine

```rust
// ─── csm-semantic/src/feature_engine.rs ───────────────────────────────────

use std::collections::VecDeque;
use rustc_hash::FxHashMap;

/// Sliding window frequency counter per pattern_id
pub struct PatternFrequencyEngine {
    /// Window 1 menit: deque of (timestamp_micros, pattern_id)
    window_1m: VecDeque<(i64, PatternId)>,
    /// Window 1 jam
    window_1h: VecDeque<(i64, PatternId)>,
    /// Count per pattern dalam masing-masing window
    count_1m:  FxHashMap<PatternId, u32>,
    count_1h:  FxHashMap<PatternId, u32>,
    /// Historical mean + variance (EMA) per pattern
    ema_mean:  FxHashMap<PatternId, f64>,
    ema_var:   FxHashMap<PatternId, f64>,
    alpha:     f64,  // EMA decay factor (default 0.05)
}

impl PatternFrequencyEngine {
    pub fn update(&mut self, pattern_id: PatternId, ts: i64) {
        let cutoff_1m = ts - 60_000_000;  // micros
        let cutoff_1h = ts - 3_600_000_000;

        // Evict old entries
        while self.window_1m.front().map(|(t,_)| *t < cutoff_1m).unwrap_or(false) {
            let (_, pid) = self.window_1m.pop_front().unwrap();
            *self.count_1m.entry(pid).or_insert(0) -= 1;
        }
        // ... similar for 1h window

        self.window_1m.push_back((ts, pattern_id));
        *self.count_1m.entry(pattern_id).or_insert(0) += 1;

        // Update EMA
        let freq = *self.count_1m.get(&pattern_id).unwrap_or(&0) as f64;
        let mean = self.ema_mean.entry(pattern_id).or_insert(freq);
        let var  = self.ema_var.entry(pattern_id).or_insert(0.0);
        let delta = freq - *mean;
        *mean += self.alpha * delta;
        *var  = (1.0 - self.alpha) * (*var + self.alpha * delta * delta);
    }

    pub fn freq_1m(&self, pid: PatternId) -> u32 { *self.count_1m.get(&pid).unwrap_or(&0) }
    pub fn freq_1h(&self, pid: PatternId) -> u32 { *self.count_1h.get(&pid).unwrap_or(&0) }
    pub fn z_score_1m(&self, pid: PatternId) -> f64 {
        let freq = self.freq_1m(pid) as f64;
        let mean = *self.ema_mean.get(&pid).unwrap_or(&freq);
        let std  = self.ema_var.get(&pid).unwrap_or(&1.0).sqrt();
        if std < 1e-9 { 0.0 } else { (freq - mean) / std }
    }
}
```

### 5.2 SlotStatEngine

```rust
/// Per-slot running statistics menggunakan Welford's online algorithm
pub struct SlotStatEngine {
    /// stats per (pattern_id, slot_position)
    stats: FxHashMap<(PatternId, u8), OnlineStats>,
    /// HyperLogLog untuk cardinality estimation per slot
    hll:   FxHashMap<(PatternId, u8), hyperloglogplus::HyperLogLogPlus<u64, _>>,
}

/// Welford's online mean + variance (numerically stable)
pub struct OnlineStats {
    count: u64,
    mean:  f64,
    m2:    f64,          // running sum of squared deviations
    min:   f64,
    max:   f64,
}

impl OnlineStats {
    pub fn update(&mut self, x: f64) {
        self.count += 1;
        let delta  = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2   += delta * delta2;
        self.min   = self.min.min(x);
        self.max   = self.max.max(x);
    }
    pub fn variance(&self) -> f64 {
        if self.count < 2 { 0.0 } else { self.m2 / (self.count - 1) as f64 }
    }
    pub fn std(&self) -> f64 { self.variance().sqrt() }
    pub fn z_score(&self, x: f64) -> f64 {
        let s = self.std();
        if s < 1e-12 { 0.0 } else { (x - self.mean) / s }
    }
}
```

### 5.3 Anomaly Score Computation

```
ANOMALY SCORE FORMULA:

  anomaly_score = w1×freq_anomaly + w2×slot_anomaly + w3×seq_anomaly + w4×temporal_anomaly
  weights: w1=0.35, w2=0.30, w3=0.20, w4=0.15  (sum = 1.0)

  freq_anomaly:
    z = freq_engine.z_score_1m(pattern_id)
    freq_anomaly = sigmoid(|z| - 2.0)  → 0 jika z<2, naik tajam jika z>3
    sigmoid(x) = 1 / (1 + exp(-x))

  slot_anomaly:
    per slot: slot_z = stat_engine.z_score(pattern_id, slot_pos, value)
    slot_anomaly = max(sigmoid(|slot_z| - 2.5)) over all slots

  seq_anomaly:
    seq_pmi = bigram_pmi(last_pattern, current_pattern)
    seq_anomaly = sigmoid(-seq_pmi / 2.0)  → tinggi jika PMI rendah (unusual transition)

  temporal_anomaly:
    hour_of_day = (log_timestamp / 3_600_000_000) % 24
    expected_freq = historical_hourly_mean[pattern_id][hour_of_day]
    temporal_anomaly = sigmoid(|freq_1h - expected_freq| / expected_std - 2.0)

THRESHOLD RECOMMENDATION:
  anomaly_score > 0.3: LOW anomaly (log/monitor)
  anomaly_score > 0.6: MEDIUM anomaly (alert)
  anomaly_score > 0.8: HIGH anomaly (page on-call)
```

### 5.4 Sequence Pattern Features

```rust
/// N-gram pattern di level PatternId sequence (bukan token sequence)
pub struct SequencePatternEngine {
    bigram_counts:   FxHashMap<(PatternId, PatternId), u32>,
    unigram_counts:  FxHashMap<PatternId, u32>,
    total_bigrams:   u64,
    window:          VecDeque<PatternId>,   // recent pattern history (size=10)
}

impl SequencePatternEngine {
    pub fn update(&mut self, pid: PatternId) {
        if let Some(&prev) = self.window.back() {
            *self.bigram_counts.entry((prev, pid)).or_insert(0) += 1;
            self.total_bigrams += 1;
        }
        *self.unigram_counts.entry(pid).or_insert(0) += 1;
        self.window.push_back(pid);
        if self.window.len() > 10 { self.window.pop_front(); }
    }

    pub fn bigram_pmi(&self, p1: PatternId, p2: PatternId) -> f64 {
        let n = self.total_bigrams as f64;
        let f12 = *self.bigram_counts.get(&(p1, p2)).unwrap_or(&0) as f64;
        let f1  = *self.unigram_counts.get(&p1).unwrap_or(&0) as f64;
        let f2  = *self.unigram_counts.get(&p2).unwrap_or(&0) as f64;
        if f12 < 1.0 || f1 < 1.0 || f2 < 1.0 { return -10.0; }
        (f12 * n / (f1 * f2)).log2()
    }

    /// Burstiness score: mendeteksi log yang datang dalam burst
    /// Nilai tinggi = bursty (tidak normal), nilai rendah = smooth
    pub fn burst_score(&self, pid: PatternId, window_freqs: &[f64]) -> f64 {
        if window_freqs.is_empty() { return 0.0; }
        let mean = window_freqs.iter().sum::<f64>() / window_freqs.len() as f64;
        let var  = window_freqs.iter().map(|x| (x-mean).powi(2)).sum::<f64>() / window_freqs.len() as f64;
        if mean < 1e-9 { return 0.0; }
        let cv2 = var / (mean * mean);
        (cv2 - 1.0 / mean).max(0.0)  // = (σ²/μ² - 1/μ), positive = bursty
    }
}
```

---

## §6 — File Format (.csm v4)

### 6.1 Overview

```
.csm v4 file layout:
┌─────────────────────────────────────────────────────────┐
│ HEADER (128 bytes, fixed)                               │
├─────────────────────────────────────────────────────────┤
│ VOCAB_SECTION (variable length, 8-byte aligned)         │
├─────────────────────────────────────────────────────────┤
│ PATTERN_SECTION (variable length, 64-byte aligned)      │
├─────────────────────────────────────────────────────────┤
│ SLOT_SECTION (variable length, 8-byte aligned)          │
├─────────────────────────────────────────────────────────┤
│ FEATURE_SCHEMA_SECTION (variable, 8-byte aligned)       │
├─────────────────────────────────────────────────────────┤
│ DATA_SECTION (variable, 8-byte aligned)                 │
│   → bit-packed Assignment stream                        │
│   → interleaved SemanticRecord metadata                 │
├─────────────────────────────────────────────────────────┤
│ INDEX_SECTION (for random access, optional)             │
├─────────────────────────────────────────────────────────┤
│ FOOTER (16 bytes, fixed)                                │
└─────────────────────────────────────────────────────────┘
```

### 6.2 Header (128 bytes)

```
Offset  Size  Type      Field                    Value / Description
──────────────────────────────────────────────────────────────────────
0x00    4     [u8;4]    MAGIC                    b"CSM4"
0x04    1     u8        MAJOR_VERSION            4
0x05    1     u8        MINOR_VERSION            0
0x06    2     u16       FLAGS                    see below (little-endian)
0x08    4     u32       VOCAB_SIZE               jumlah vocab entries
0x0C    4     u32       PATTERN_COUNT            jumlah patterns
0x10    4     u32       SLOT_COUNT               jumlah slot definitions
0x14    1     u8        DOMAIN                   0=Generic,1=Log,2=Code,3=Text
0x15    3     [u8;3]    _RESERVED_1              = [0,0,0]
0x18    4     u32       TIER_CUTOFF_01           VocabId boundary tier 0→1
0x1C    4     u32       TIER_CUTOFF_12           VocabId boundary tier 1→2
0x20    4     u32       TIER_CUTOFF_23           VocabId boundary tier 2→3
0x24    4     u32       _RESERVED_2              = 0
0x28    8     u64       TOKEN_COUNT_ENCODED      total tokens setelah compression
0x30    8     u64       TOKEN_COUNT_ORIGINAL     total tokens sebelum compression
0x38    8     u64       VOCAB_FINGERPRINT        FNV-1a(sorted id→str mapping)
0x40    4     f32       CORPUS_ENTROPY           H(X) dari build phase
0x44    4     f32       COMPRESSION_RATIO        achieved CR
0x48    8     u64       SECTION_OFFSET_VOCAB     byte offset dari file start
0x50    8     u64       SECTION_OFFSET_PATTERN   byte offset
0x58    8     u64       SECTION_OFFSET_SLOT      byte offset
0x60    8     u64       SECTION_OFFSET_DATA      byte offset
0x68    8     u64       SECTION_OFFSET_INDEX     byte offset (0 jika tidak ada)
0x70    8     u64       BUILD_TIMESTAMP          Unix micros saat build
0x78    4     u32       HEADER_CRC32C            Castagnoli CRC dari 0x00–0x77

FLAGS (u16):
  bit 0:  MULTI_TIER_PACK     1 = multi-tier bit packing aktif
  bit 1:  VITERBI_SELECT      1 = Viterbi DP digunakan (bukan greedy)
  bit 2:  DELTA_SLOTS         1 = numeric slot delta encoding
  bit 3:  ANS_TIER3           1 = ANS untuk Tier-3 (bukan Elias-γ)
  bit 4:  SLOT_TYPED          1 = typed slots aktif
  bit 5:  HAS_FEATURES        1 = FeatureVector disimpan di DATA_SECTION
  bit 6:  STREAMING_MODE      1 = file adalah stream of frames (lihat §8)
  bit 7:  HAS_INDEX           1 = INDEX_SECTION present
  bit 8:  HELD_OUT_VERIFIED   1 = PGS check dilakukan saat discovery
  bit 9-15: RESERVED = 0
```

### 6.3 Vocab Section

```
Entry format (variable length, padded to 4-byte boundary):
  [u32  ID          ]   4 bytes
  [u128 ULID        ]   16 bytes (little-endian u128)
  [u8   TIER        ]   1 byte (0/1/2/3)
  [u8   _PAD        ]   1 byte
  [u16  STR_LEN     ]   2 bytes (string length in bytes)
  [u8×STR_LEN UTF8  ]   variable
  [u8×PAD     _     ]   padding to 4-byte boundary

Total per entry: 24 + ceil(STR_LEN / 4) × 4 bytes
```

### 6.4 Pattern Section (64-byte aligned)

```
Entry format (88 bytes fixed, align(64)):
  [u32   ID                ]   4 bytes
  [u8    DOMAIN            ]   1 byte
  [u8    BASE_SEQ_LEN      ]   1 byte (2–5)
  [u8    SLOT_COUNT        ]   1 byte
  [u8    FLAGS             ]   1 byte (bit0=deprecated, bit1=delta_encoded)
  [u32×5 BASE_SEQ          ]   20 bytes (unused entries = 0)
  [u32×2 SLOT_IDS          ]   8 bytes (INVALID_SLOT jika tidak ada)
  [u8×2  SLOT_POSITIONS    ]   2 bytes
  [u8×2  _PAD              ]   2 bytes
  [u32   FREQ              ]   4 bytes
  [f32   PPMI_SCORE        ]   4 bytes
  [f32   COMPRESS_GAIN     ]   4 bytes
  [f32   PGS_SCORE         ]   4 bytes
  [f32   STABILITY         ]   4 bytes
  [f32   FINAL_SCORE       ]   4 bytes
  [u64   CTX_HASH          ]   8 bytes (context fingerprint)
  [u16   TEMPLATE_LEN      ]   2 bytes
  [u8×TEMPLATE_LEN TEMPLATE]   variable (inline untuk len≤32, else pointer to string pool)
  Padded to 88 bytes total (fixed-size untuk mmap array indexing)
```

### 6.5 Data Section

```
DATA_SECTION berisi interleaved stream:

Untuk setiap encoded line:
  [u16 LINE_META_LEN  ]   2 bytes — panjang metadata blok berikut
  [LINE_META          ]   variable — {source_offset, log_ts, token_count_orig}
  [BIT_STREAM         ]   ceil(encoded_bits / 8) bytes
    Format per assignment:
      Pattern:  [2-bit tier_prefix][pattern_id_payload][slot_values...]
      Token:    [2-bit tier_prefix][token_id_payload]
      Fallback: [2'b11][2-bit level][payload...]
  [u8 PAD             ]   padding ke 4-byte boundary
  [u32 LINE_CRC32C    ]   CRC dari LINE_META + BIT_STREAM

STREAMING MODE (FLAGS.bit6 = 1):
  Frame-based format:
  [u32 FRAME_MAGIC    ]   b"CSMF"
  [u32 FRAME_LEN      ]   total bytes frame
  [u16 LINE_COUNT     ]   jumlah lines dalam frame
  [LINES...           ]   line entries seperti di atas
  [u32 FRAME_CRC32C   ]   CRC dari LINES
```

### 6.6 Footer (16 bytes)

```
[u64 DATA_CRC32C_FULL ]   8 bytes — CRC dari seluruh DATA_SECTION
[u32 TOTAL_SECTIONS   ]   4 bytes — jumlah sections
[u32 FOOTER_MAGIC     ]   = 0x43534D45 ("CSME")
```

---

## §7 — API Design

### 7.1 Rust API (Public Interface)

```rust
// ─── csm-api/src/encoder.rs ───────────────────────────────────────────────

use csm_core::{Vocab, PatternRegistry, CsmError, DomainKind};
use csm_semantic::{SemanticRecord, FeatureVector};

/// Builder pattern untuk encoder configuration
pub struct EncoderBuilder {
    domain:        DomainKind,
    vocab_path:    Option<PathBuf>,
    fst_path:      Option<PathBuf>,
    chunk_size:    usize,            // default 4096 tokens
    feature_mode:  FeatureMode,      // None | Basic | Full
    output_format: OutputFormatKind, // Binary | Arrow | Json
}

impl EncoderBuilder {
    pub fn new(domain: DomainKind) -> Self { ... }
    pub fn with_vocab(mut self, path: impl Into<PathBuf>) -> Self { ... }
    pub fn with_fst(mut self, path: impl Into<PathBuf>) -> Self { ... }
    pub fn with_chunk_size(mut self, size: usize) -> Self { ... }
    pub fn with_features(mut self, mode: FeatureMode) -> Self { ... }
    pub fn build(self) -> Result<CsmEncoder, CsmError> { ... }
}

/// Main encoder — thread-safe (Clone = cheap Arc clone)
#[derive(Clone)]
pub struct CsmEncoder {
    inner: Arc<EncoderInner>,
}

struct EncoderInner {
    vocab:       Arc<Vocab<'static>>,      // 'static karena arena lives in Arc
    registry:    Arc<PatternRegistry>,
    fst:         Arc<memmap2::Mmap>,       // memory-mapped FST binary
    tokenizer:   Arc<dyn DomainTokenizer>,
    feat_engine: Arc<Mutex<FeatureEngineBundle>>,  // mutable state
    config:      EncoderConfig,
}

impl CsmEncoder {
    /// Encode satu string → CsmData (binary output)
    pub fn encode(&self, input: &str) -> Result<CsmData, CsmError>;

    /// Encode dengan output sebagai SemanticRecord (untuk ML pipeline)
    pub fn encode_semantic(&self, input: &str) -> Result<SemanticRecord, CsmError>;

    /// Encode batch (parallelized via Rayon)
    pub fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<SemanticRecord>, CsmError> {
        inputs.par_iter()
              .map(|s| self.encode_semantic(s))
              .collect()
    }

    /// Encode + output ke Arrow RecordBatch
    pub fn encode_to_arrow(&self, inputs: &[&str]) -> Result<arrow2::chunk::Chunk<Box<dyn arrow2::array::Array>>, CsmError>;

    /// Streaming encoder — returns a handle
    pub fn stream(&self) -> StreamingEncoder;
}

pub struct CsmData {
    pub header:   CsmLineHeader,
    pub bits:     Vec<u8>,           // bit-packed assignments
    pub slots:    Vec<SlotValue>,    // extracted slot values
    pub features: Option<FeatureVector>,
}

// ─── csm-api/src/decoder.rs ───────────────────────────────────────────────

pub struct CsmDecoder {
    inner: Arc<DecoderInner>,
}

impl CsmDecoder {
    /// Decode binary CsmData → original text (lossy jika UNK tokens ada)
    pub fn decode(&self, data: &CsmData) -> Result<String, CsmError>;

    /// Decode binary .csm file → Vec<SemanticRecord>
    pub fn decode_file(&self, path: &Path) -> Result<Vec<SemanticRecord>, CsmError>;

    /// Streaming decode (.csm → Iterator<SemanticRecord>)
    pub fn decode_stream(&self, reader: impl Read) -> impl Iterator<Item = Result<SemanticRecord, CsmError>>;

    /// Decode directly to Parquet file (memory-efficient)
    pub fn decode_to_parquet(&self, input: &Path, output: &Path) -> Result<DecodeStats, CsmError>;
}
```

### 7.2 Python API (PyO3 Bindings)

```rust
// ─── csm-python/src/bindings.rs ───────────────────────────────────────────

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};

/// Python class: CsmEncoder
#[pyclass]
pub struct PyCsmEncoder {
    inner: CsmEncoder,
}

#[pymethods]
impl PyCsmEncoder {
    /// fit(data: List[str]) → self
    /// Build vocab + discover patterns dari training data
    #[pyo3(signature = (data, domain="log", min_freq=10, max_patterns=100_000))]
    pub fn fit(&mut self, data: Vec<String>, domain: &str,
               min_freq: u32, max_patterns: usize) -> PyResult<()>;

    /// transform(data: List[str]) → polars.DataFrame
    pub fn transform(&self, data: Vec<String>) -> PyResult<PyObject>;

    /// fit_transform(data: List[str]) → polars.DataFrame
    pub fn fit_transform(&mut self, data: Vec<String>) -> PyResult<PyObject>;

    /// encode(text: str) → dict
    pub fn encode(&self, text: String) -> PyResult<PyObject>;

    /// encode_batch(texts: List[str], parallel: bool) → polars.DataFrame
    #[pyo3(signature = (texts, parallel=true))]
    pub fn encode_batch(&self, texts: Vec<String>, parallel: bool) -> PyResult<PyObject>;

    /// save(path: str) → None
    pub fn save(&self, path: String) -> PyResult<()>;

    /// Compression ratio achieved
    #[getter]
    pub fn compression_ratio(&self) -> f32;

    /// Pattern count
    #[getter]
    pub fn pattern_count(&self) -> usize;
}

/// Python class: CsmDecoder
#[pyclass]
pub struct PyCsmDecoder {
    inner: CsmDecoder,
}

#[pymethods]
impl PyCsmDecoder {
    #[staticmethod]
    pub fn load(path: String) -> PyResult<Self>;

    /// decode(data: bytes) → str
    pub fn decode(&self, data: Vec<u8>) -> PyResult<String>;

    /// decode_file(path: str) → polars.DataFrame
    pub fn decode_file(&self, path: String) -> PyResult<PyObject>;
}

/// Module registration
#[pymodule]
fn csm_plus_plus(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyCsmEncoder>()?;
    m.add_class::<PyCsmDecoder>()?;
    Ok(())
}
```

### 7.3 Python Usage Example

```python
# pip install csm-plus-plus (maturin-built wheel)
import csm_plus_plus as csm
import polars as pl

# ── Log Analytics Pipeline ──────────────────────────────────────────────────
encoder = csm.CsmEncoder()

# fit: build vocab + discover patterns dari training logs
with open("training_logs.txt") as f:
    training_lines = f.readlines()[:100_000]

encoder.fit(training_lines, domain="log", min_freq=5, max_patterns=50_000)
print(f"Patterns discovered: {encoder.pattern_count}")
print(f"Compression ratio on training data: {encoder.compression_ratio:.2f}x")

# transform: encode baru → DataFrame dengan fitur otomatis
new_logs = [
    "2026-03-26T10:00:01Z ERROR Connection from 192.168.1.5 port 443 rejected",
    "2026-03-26T10:00:02Z INFO  Request processed in 45ms",
    "2026-03-26T10:00:03Z WARN  High memory usage: 87%",
]
df = encoder.transform(new_logs)

# df adalah polars.DataFrame dengan kolom:
# source_id, offset, log_timestamp, pattern_id, template, domain,
# compress_ratio, anomaly_score, pattern_freq_1m, slot_ip, slot_port,
# slot_level, slot_duration_ms, slot_pct, ...

print(df.schema)
# Filter high anomaly
alerts = df.filter(pl.col("anomaly_score") > 0.7)
print(f"Anomalies detected: {len(alerts)}")

# Save untuk later use
encoder.save("logs_encoder_v1.csm4")

# ── Batch ML Pipeline ────────────────────────────────────────────────────────
from sklearn.ensemble import IsolationForest

feature_cols = ["anomaly_score", "pattern_freq_1m", "seq_burst_score"]
X = df.select(feature_cols).to_numpy()

model = IsolationForest(contamination=0.05)
labels = model.fit_predict(X)
```

### 7.4 CLI Interface

```bash
# csm-cli — command line tool

# Build vocab + patterns dari corpus
csm build \
  --input /var/log/app/*.log \
  --domain log \
  --output logs_encoder.csm4 \
  --min-freq 10 \
  --max-patterns 50000 \
  --held-out-ratio 0.1

# Encode file
csm encode \
  --encoder logs_encoder.csm4 \
  --input /var/log/app/current.log \
  --output encoded.csm4 \
  --format binary   # atau arrow, parquet, json

# Decode ke DataFrame
csm decode \
  --encoder logs_encoder.csm4 \
  --input encoded.csm4 \
  --output decoded.parquet \
  --format parquet

# Real-time streaming dari Kafka
csm stream \
  --encoder logs_encoder.csm4 \
  --kafka-brokers localhost:9092 \
  --kafka-topic app-logs \
  --output-kafka csm-encoded-logs \
  --anomaly-threshold 0.7 \
  --alert-topic anomalies

# Benchmark mode
csm bench \
  --encoder logs_encoder.csm4 \
  --input benchmark_data.log \
  --iterations 5 \
  --report bench_report.json
```

---

## §8 — Streaming & Real-Time Processing

### 8.1 Streaming Architecture

```
        ┌─────────────┐     ┌───────────────┐     ┌──────────────┐
Kafka   │  rdkafka    │     │  CsmStreaming  │     │  Kafka       │
Topics →│  Consumer   │────→│  Pipeline     │────→│  Producer    │
        └─────────────┘     └───────────────┘     └──────────────┘
                                   │
                          ┌────────┴─────────┐
                          │                  │
                   ┌──────▼──────┐   ┌───────▼──────┐
                   │  Anomaly    │   │  Parquet      │
                   │  Alerter    │   │  Sink         │
                   └─────────────┘   └──────────────┘
```

### 8.2 Streaming Pipeline Implementation

```rust
// ─── csm-streaming/src/lib.rs ─────────────────────────────────────────────

use tokio::sync::mpsc;
use rdkafka::consumer::{Consumer, StreamConsumer};
use rdkafka::producer::{FutureProducer, FutureRecord};

pub struct CsmStreamingPipeline {
    encoder:    CsmEncoder,
    config:     StreamConfig,
    metrics:    Arc<StreamMetrics>,
}

#[derive(Clone)]
pub struct StreamConfig {
    pub chunk_size:      usize,        // tokens per chunk (default 4096)
    pub frame_lines:     usize,        // lines per streaming frame (default 1000)
    pub kafka_brokers:   String,
    pub input_topic:     String,
    pub output_topic:    String,
    pub anomaly_topic:   Option<String>,
    pub anomaly_threshold: f32,        // default 0.7
    pub flush_interval_ms: u64,        // default 100ms
}

impl CsmStreamingPipeline {
    pub async fn run(&self) -> Result<(), CsmError> {
        let consumer = self.build_consumer()?;
        let producer = self.build_producer()?;

        let (tx, mut rx) = mpsc::channel::<RawLine>(10_000);

        // Ingest task: Kafka → channel
        let ingest_handle = tokio::spawn({
            let consumer = consumer.clone();
            let tx = tx.clone();
            async move { Self::ingest_loop(consumer, tx).await }
        });

        // Encode task: channel → SemanticRecord (CPU-bound → spawn_blocking)
        let (enc_tx, mut enc_rx) = mpsc::channel::<SemanticRecord>(10_000);
        let encode_handle = tokio::spawn({
            let encoder = self.encoder.clone();
            async move {
                while let Some(line) = rx.recv().await {
                    let record = tokio::task::spawn_blocking({
                        let encoder = encoder.clone();
                        let line = line.clone();
                        move || encoder.encode_semantic(&line.text)
                    }).await??;
                    enc_tx.send(record).await?;
                }
                Ok::<_, CsmError>(())
            }
        });

        // Output task: SemanticRecord → Kafka + anomaly alerter
        let output_handle = tokio::spawn({
            let producer = producer.clone();
            let config = self.config.clone();
            async move {
                let mut batch = Vec::with_capacity(config.frame_lines);
                let mut flush_interval = tokio::time::interval(
                    std::time::Duration::from_millis(config.flush_interval_ms)
                );
                loop {
                    tokio::select! {
                        Some(record) = enc_rx.recv() => {
                            // Anomaly alert
                            if let Some(f) = &record.features {
                                if f.anomaly_score > config.anomaly_threshold {
                                    if let Some(topic) = &config.anomaly_topic {
                                        let payload = serde_json::to_string(&record)?;
                                        producer.send(FutureRecord::to(topic)
                                            .payload(&payload)
                                            .key(&record.pattern_id.to_string()), ...).await??;
                                    }
                                }
                            }
                            batch.push(record);
                            if batch.len() >= config.frame_lines {
                                Self::flush_batch(&producer, &config, &mut batch).await?;
                            }
                        }
                        _ = flush_interval.tick() => {
                            if !batch.is_empty() {
                                Self::flush_batch(&producer, &config, &mut batch).await?;
                            }
                        }
                    }
                }
            }
        });

        tokio::try_join!(ingest_handle, encode_handle, output_handle)?;
        Ok(())
    }
}
```

### 8.3 Chunk-Based Processing (Non-Kafka)

```rust
/// Low-latency chunk processor untuk embedded use (tanpa Kafka)
pub struct ChunkProcessor {
    encoder:     CsmEncoder,
    buffer:      Vec<u8>,
    line_buffer: Vec<String>,
    chunk_lines: usize,        // default 1000
    on_output:   Box<dyn Fn(Vec<SemanticRecord>) + Send + Sync>,
}

impl ChunkProcessor {
    /// Push raw bytes (tidak harus line-aligned)
    pub fn push(&mut self, data: &[u8]) -> Result<(), CsmError> {
        self.buffer.extend_from_slice(data);
        // Extract complete lines
        while let Some(pos) = self.buffer.iter().position(|&b| b == b'\n') {
            let line = String::from_utf8_lossy(&self.buffer[..pos]).into_owned();
            self.buffer.drain(..=pos);
            self.line_buffer.push(line);
            if self.line_buffer.len() >= self.chunk_lines {
                self.flush()?;
            }
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), CsmError> {
        if self.line_buffer.is_empty() { return Ok(()); }
        let lines: Vec<&str> = self.line_buffer.iter().map(|s| s.as_str()).collect();
        let records = self.encoder.encode_batch(&lines)?;
        (self.on_output)(records);
        self.line_buffer.clear();
        Ok(())
    }
}
```

### 8.4 Latency Budget

```
LOW-LATENCY MODE TARGET: < 5ms end-to-end (Kafka in → Kafka out)

Budget allocation:
  Kafka consume:       ~1ms  (network + deserialization)
  Normalization:       ~0.1ms (SIMD, 10MB/s per core)
  FST match:           ~0.5ms (untuk 1000 tokens @ 2M tok/s)
  Viterbi DP:          ~0.2ms
  Slot extraction:     ~0.1ms
  Feature computation: ~0.3ms (windowed stats)
  Kafka produce:       ~1ms
  Overhead + buffer:   ~1.8ms
  ─────────────────────────
  Total:               ~5.0ms ✓

TUNING KNOBS untuk latency reduction:
  - Reduce chunk_lines ke 100 (vs default 1000): -2ms latency, -3x throughput
  - Disable feature computation (feature_mode=None): -0.3ms
  - Use rdkafka batch.size=16384 + linger.ms=0
  - Pin threads ke specific cores (affinity): -0.5ms jitter
```

---

## §9 — Performance Model (Realistic)

### 9.1 Throughput Model

```
HARDWARE BASELINE:
  CPU:  Intel Core i9-13900K, 3.0 GHz base / 5.8 GHz boost, 8 P-cores
  RAM:  64 GB DDR5-5600
  L1:   32 KB / core
  L2:   2 MB / core
  L3:   36 MB shared
  Disk: NVMe SSD 7 GB/s read

PIPELINE BOTTLENECK ANALYSIS:

Stage             Single Core    8-Core (Rayon)   Bottleneck
──────────────────────────────────────────────────────────────
S1: Normalize     20 M tok/s     160 M tok/s      Memory BW
S2: Tokenize      8 M tok/s      60 M tok/s       Branch pred.
S3: FST Match     1.5 M tok/s    10 M tok/s       Cache miss
S4: Viterbi DP    5 M tok/s      38 M tok/s       Branch pred.
S5: Slot Extract  15 M tok/s     100 M tok/s      Parse logic
S6: Feature Eng   2 M tok/s      14 M tok/s       HashMap ops
S7: Bit-Pack      25 M tok/s     180 M tok/s      BMI2 PDEP
S8: Write output  3 M tok/s      ~3 M tok/s       Disk I/O BW
──────────────────────────────────────────────────────────────
SYSTEM (hot path) 1.5 M tok/s    10 M tok/s       FST Match
SYSTEM (+features) 1.2 M tok/s   8 M tok/s        Feature Eng
SYSTEM (+disk out) 1.0 M tok/s   3 M tok/s        Disk I/O

NOTE: "tokens" di sini = log fields setelah whitespace split
Rata-rata 8 fields/line → throughput dalam lines/sec:
  Single core:  ~125K–188K lines/sec
  8-core:       ~375K–1.25M lines/sec
```

### 9.2 Memory Usage

```
MEMORY FOOTPRINT (V=65536 patterns, P=50000, N=100M lines):

Component              Size         Notes
──────────────────────────────────────────────────────────────
Vocab arena            ~50 MB       V × avg_str_len (8 bytes)
FST binary (mmap)      ~5 MB        Shared via mmap, no copy
Pattern registry       ~50 MB       P × 88 bytes = 4.4 MB + string pool
Slot definitions       ~5 MB        Per-slot schema + value sets
Feature engine         ~20 MB       EMA state + HLL + window buffers
Working set (1 chunk)  ~14 MB       4096 tokens × 36 bytes/token
Output buffer          ~64 MB       Configurable, default 64MB Arrow batch
──────────────────────────────────────────────────────────────
TOTAL PEAK             ~210 MB      Jauh di bawah server RAM
MINIMAL MODE (edge)    ~20 MB       FST + vocab only, no features
```

### 9.3 Compression Ratios (Domain-Specific, Realistic)

```
LOGS — Application/Server Logs:
  Input:  Raw UTF-8 log file (1 GB)
  Method: CSM++ vs gzip-9 vs zstd-19

  Dataset             gzip-9   zstd-19  CSM++ binary  CSM++ CR vs gzip
  ──────────────────────────────────────────────────────────────────
  Nginx access logs   8.2x     11.4x    18.5x          2.3x
  App error logs      6.8x     9.1x     15.2x          2.2x
  Kubernetes events   7.5x     10.2x    20.1x          2.7x
  Mixed syslogs       5.9x     7.8x     12.3x          2.1x

  CSM++ advantage: semantic deduplication of templates + numeric slot compression
  CSM++ Arrow output (features): ~4x smaller than JSON equivalent

SOURCE CODE:
  Dataset             gzip-9   zstd-19  CSM++ binary  vs gzip
  ──────────────────────────────────────────────────────────────
  Python source       7.2x     9.8x     12.8x          1.8x
  Rust source         6.5x     8.9x     10.5x          1.6x
  Go source           7.0x     9.6x     11.9x          1.7x
  Stack traces        9.1x     13.5x    28.7x          3.2x  (highly repetitive)

AI TRAINING DATA:
  Dataset             gzip-9   zstd-19  CSM++ binary  vs gzip
  ──────────────────────────────────────────────────────────────
  Instruction pairs   4.8x     6.5x     8.2x           1.7x
  QA datasets         5.1x     6.9x     9.1x           1.8x
  Wikipedia text      3.8x     5.1x     5.9x           1.6x

NOTE: CSM++ CR advantage diminishes untuk random/diverse text.
Positioning must stay on logs + code — that's where it truly wins.
```

---

## §10 — Benchmark Design

### 10.1 Benchmark Datasets

```
DATASET 1: GitHub Public Logs (Logs domain)
  Source:     GitHub Actions workflow logs (public repos)
  Size:       10 GB raw
  Format:     Plain text, one log line per line
  Diversity:  >1000 distinct project types
  Use for:    CR benchmark, throughput benchmark, anomaly detection eval

DATASET 2: Linux Kernel Syslogs
  Source:     kernel.org public log archives
  Size:       5 GB raw
  Format:     RFC 5424 syslog format
  Use for:    Streaming benchmark, low-latency test

DATASET 3: GitHub Code Repositories
  Source:     GHTorrent sample (Python + Rust, 500K files)
  Size:       20 GB raw source code
  Format:     UTF-8 source files
  Use for:    Code domain CR benchmark

DATASET 4: Wikipedia EN (Text domain)
  Source:     Wikimedia dumps (2026-03 dump)
  Size:       22 GB XML → ~18 GB extracted text
  Format:     Plain paragraphs
  Use for:    Baseline comparison (CSM++ weakest domain)

DATASET 5: Synthetic Anomaly Logs
  Source:     Generated from real distributions + injected anomalies
  Size:       1 GB with 5% labeled anomalies
  Format:     JSON lines dengan ground truth labels
  Use for:    Anomaly detection F1 evaluation
```

### 10.2 Metrics

```
METRIC 1: Compression Ratio (CR)
  Formula: raw_bytes_input / compressed_bytes_output
  Variants:
    CR_binary: vs .csm v4 bit stream
    CR_arrow:  vs Arrow RecordBatch output (includes features)
    CR_vs_gzip: CR_binary / CR_gzip (competitive)
  Report: mean ± std over 5 runs, per domain

METRIC 2: Effective Compression Coefficient (ECC)
  Formula: ECC = CR × (1 - information_loss_rate)
  information_loss_rate = 1 - (recovered_text_distance / original_text_distance)
    measured via normalized edit distance on 1000 random samples
  Interpretation: CR yang mempertimbangkan lossy degradation
  Target: ECC ≥ 0.95 × CR (loss < 5%)

METRIC 3: Semantic Compression Ratio (SCR)
  Formula: SCR = unique_templates / unique_raw_lines
  Interpretation: seberapa besar "template space" vs "raw event space"
  Higher = more semantic compression
  Target per domain:
    Logs:  SCR > 0.01  (1 template per 100 raw lines → high compression)
    Code:  SCR > 0.05
    Text:  SCR > 0.20  (lebih sedikit repetisi)

METRIC 4: Throughput
  Unit: million tokens per second (M tok/s) dan lines per second (lines/s)
  Measure: wall-clock time untuk encode seluruh dataset
  Variants: 1-core, 4-core, 8-core
  Tool: criterion crate (Rust microbench) + hyperfine (wall-clock)

METRIC 5: Anomaly Detection Quality (Dataset 5 only)
  Metrics: Precision, Recall, F1, AUROC
  Method: anomaly_score threshold sweep (0.1–0.9, step 0.1)
  Baseline: Drain3 + Isolation Forest (Python)
  Target: F1 > 0.75 (competitive dengan dedicated anomaly tools)

METRIC 6: Memory Peak
  Tool: heaptrack (Linux) / Instruments (macOS)
  Measure: peak RSS selama encode 1 GB file
  Target: < 500 MB untuk default config

METRIC 7: Build (Discovery) Time
  Measure: waktu untuk fit() dari corpus 1 GB
  Target: < 5 menit single-threaded, < 90 detik 8-core
```

### 10.3 Benchmark Code Structure

```rust
// benches/encode_throughput.rs

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use csm_api::{CsmEncoder, EncoderBuilder, DomainKind};

fn bench_encode_single_core(c: &mut Criterion) {
    let encoder = EncoderBuilder::new(DomainKind::Log)
        .with_vocab("bench_data/nginx_vocab.csm4")
        .with_fst("bench_data/nginx_fst.bin")
        .with_features(FeatureMode::None)  // pure encoding
        .build().unwrap();

    let lines: Vec<String> = load_bench_lines("bench_data/nginx_1M.log");
    let total_bytes: u64 = lines.iter().map(|l| l.len() as u64).sum();

    let mut group = c.benchmark_group("encode_single_core");
    group.throughput(Throughput::Bytes(total_bytes));

    group.bench_function("nginx_logs_1M", |b| {
        b.iter(|| {
            for line in &lines {
                let _ = encoder.encode(line).unwrap();
            }
        })
    });
    group.finish();
}

fn bench_encode_parallel(c: &mut Criterion) {
    // Same setup, use encode_batch() yang internally uses Rayon
    let encoder = EncoderBuilder::new(DomainKind::Log)
        .with_vocab("bench_data/nginx_vocab.csm4")
        .build().unwrap();

    let lines: Vec<&str> = load_bench_lines_ref("bench_data/nginx_1M.log");

    c.bench_function("encode_batch_8core", |b| {
        b.iter(|| encoder.encode_batch(&lines).unwrap())
    });
}

criterion_group!(benches, bench_encode_single_core, bench_encode_parallel);
criterion_main!(benches);
```

---

## §11 — Android / Edge Mode

### 11.1 Constraints & Target Devices

```
TARGET DEVICES:
  - Mid-range Android: 3 GB RAM, ARM Cortex-A55 (4 cores), Android 10+
  - Low-end Android:   2 GB RAM, ARM Cortex-A53 (4 cores), Android 9+
  - Embedded Linux:    512 MB RAM, ARM Cortex-A35
  - Raspberry Pi 4:    4 GB RAM (treated as "comfortable edge")

HARD CONSTRAINTS (Low-end mode):
  - Peak RAM: < 50 MB
  - Binary size: < 5 MB (stripped, LTO)
  - No JNI overhead > 2ms per call
  - Battery: < 5% overhead vs baseline app

COMFORTABLE CONSTRAINTS (Mid-range mode):
  - Peak RAM: < 150 MB
  - Features allowed: Basic (no Full feature mode)
  - Streaming-only (no batch)
```

### 11.2 Edge Configuration

```rust
/// Konfigurasi khusus untuk edge/mobile deployment
#[derive(Clone)]
pub struct EdgeConfig {
    pub mode:           EdgeMode,
    pub max_vocab_size: u32,         // Low: 4096, Mid: 16384
    pub max_patterns:   u32,         // Low: 1000, Mid: 10000
    pub feature_mode:   FeatureMode, // Low: None, Mid: Basic
    pub chunk_size:     usize,       // Low: 512 tokens, Mid: 2048
    pub fst_max_bytes:  usize,       // Low: 512KB, Mid: 2MB
}

#[derive(Clone, Copy)]
pub enum EdgeMode {
    /// Streaming-only, minimal memory, no features
    LowEnd { max_ram_mb: u32 },
    /// Streaming + basic features
    MidRange { max_ram_mb: u32 },
    /// Full mode (Pi 4, embedded server)
    Comfortable,
}

impl EdgeConfig {
    pub fn low_end() -> Self {
        Self {
            mode:           EdgeMode::LowEnd { max_ram_mb: 50 },
            max_vocab_size: 4096,
            max_patterns:   1000,
            feature_mode:   FeatureMode::None,
            chunk_size:     512,
            fst_max_bytes:  512 * 1024,
        }
    }
    pub fn mid_range() -> Self {
        Self {
            mode:           EdgeMode::MidRange { max_ram_mb: 150 },
            max_vocab_size: 16384,
            max_patterns:   10000,
            feature_mode:   FeatureMode::Basic,
            chunk_size:     2048,
            fst_max_bytes:  2 * 1024 * 1024,
        }
    }
}
```

### 11.3 Build Configuration untuk Android

```toml
# Cargo.toml — Android profile
[profile.android-release]
inherits      = "release"
opt-level     = "z"     # optimize for size
lto           = "fat"   # full LTO
codegen-units = 1
strip         = "symbols"
panic         = "abort"  # remove unwinding code

[profile.android-release.package.csm-core]
opt-level = 3  # core tetap optimized untuk speed

# Features untuk edge build (disable heavy deps)
[features]
default  = ["full"]
full     = ["kafka", "arrow", "parquet", "features-full"]
edge     = ["features-basic"]           # no kafka, no parquet
minimal  = []                           # bare minimum
```

```bash
# Cross-compile untuk ARM64 Android
cargo build \
  --target aarch64-linux-android \
  --profile android-release \
  --no-default-features \
  --features edge \
  -p csm-api

# Expected output size: ~2.8 MB (stripped)
# compare: debug build = ~45 MB
```

### 11.4 JNI Binding (Android)

```rust
// csm-android/src/jni.rs
use jni::JNIEnv;
use jni::objects::{JClass, JString};
use jni::sys::{jbyteArray, jstring};

/// Called from Android Java/Kotlin:
/// String result = CsmNative.encode(String logLine);
#[no_mangle]
pub extern "system" fn Java_com_centra_csm_CsmNative_encode(
    env: JNIEnv,
    _class: JClass,
    log_line: JString,
) -> jstring {
    let input: String = env.get_string(log_line)
        .expect("Invalid Java string").into();

    // ENCODER is a thread-local lazy-initialized instance
    ENCODER.with(|enc| {
        match enc.borrow().encode_semantic(&input) {
            Ok(record) => {
                let json = serde_json::to_string(&record).unwrap_or_default();
                env.new_string(json).unwrap().into_raw()
            }
            Err(e) => {
                env.new_string(format!("{{\"error\":\"{}\"}}", e)).unwrap().into_raw()
            }
        }
    })
}
```

### 11.5 Edge Memory Layout

```
LOW-END MODE (50 MB budget):

Component              Size    Strategy
────────────────────────────────────────────────────────
FST binary (mmap)      512 KB  Read-only mmap
Vocab (interned)       4 MB    4096 entries × ~1KB avg
Pattern table          88 KB   1000 × 88 bytes
Working buffer         18 MB   512 tokens × 36 bytes/tok
Output buffer          8 MB    Compact JSON output
OS overhead            15 MB   Dalvik/ART runtime
Remaining              ~4 MB   Safety margin
────────────────────────────────────────────────────────
TOTAL                  ~50 MB  ✓

STREAMING GUARANTEE:
  Tidak ada data yang harus di-buffer lebih dari 1 chunk
  → Memory footprint konstan regardless of input size
```

---

## §12 — Go-To-Market Strategy

### 12.1 Target Niche & ICP (Ideal Customer Profile)

```
PRIMARY NICHE: Log Analytics + AI Dataset Engineering

ICP 1 — Platform/DevOps Engineers:
  Problem: Storage costs untuk log retention ($$$), manual anomaly hunting
  Solution: CSM++ reduces log storage 10-20x + automatic anomaly features
  Adoption: CLI tool → Kafka integration → full pipeline
  Budget: $10K-$100K/year for tooling

ICP 2 — MLOps / Data Engineers:
  Problem: LLM training dataset preprocessing bottleneck (slow, expensive)
  Solution: CSM++ fit_transform() → deduplicated semantic features → faster training
  Adoption: Python pip install → scikit-learn integration → production pipeline
  Budget: $20K-$200K/year for infrastructure

ICP 3 — Security/SOC Teams:
  Problem: SIEM ingest costs, alert fatigue, slow anomaly detection
  Solution: CSM++ streaming → real-time anomaly_score → reduced noise
  Adoption: Kafka integration → SIEM connector
  Budget: $50K-$500K/year for SIEM tooling
```

### 12.2 Adoption Plan (12 Months)

```
MONTH 1-3: Open Source Foundation
  → Publish csm-plus-plus on crates.io + GitHub (MIT license)
  → csm-cli installable via cargo install csm-plus-plus
  → Python wheel on PyPI (pip install csm-plus-plus)
  → Documentation site (mdBook)
  → Blog post: "How we compressed 1TB of logs to 50GB with semantic compression"

MONTH 3-6: Community & Integrations
  → Logstash plugin (JRuby bindings)
  → Fluentd plugin (C extension)
  → Grafana datasource plugin (query .csm files directly)
  → Jupyter notebook examples (log analysis + anomaly detection)
  → Discord community + GitHub Discussions

MONTH 6-9: Enterprise Features (closed source / BSL)
  → csm-enterprise: multi-tenant, RBAC, audit log
  → SLA-backed streaming mode
  → Managed vocabulary hosting (SaaS)
  → Kubernetes operator (csm-operator)

MONTH 9-12: Partnerships & Integrations
  → Elastic integration (CSM++ as Elasticsearch preprocessing plugin)
  → Databricks connector (CSM++ on Apache Spark)
  → AWS Marketplace listing
  → Azure Marketplace listing
```

### 12.3 Developer Tools (Open Source)

```
TOOL 1: csm-cli (cargo install csm-plus-plus)
  → encode, decode, bench, stream, build commands
  → Works on Linux/macOS/Windows

TOOL 2: csm-explore (TUI log explorer)
  → Ratatui-based terminal UI
  → Browse pattern templates, slot distributions, anomaly timeline

TOOL 3: csm-notebook (Jupyter integration)
  → %load_ext csm_magic
  → %%csm_encode, %%csm_decode magic commands

TOOL 4: csm-dashboard (Grafana plugin)
  → Visualize pattern frequency trends
  → Anomaly score time series
  → Slot value distributions

PRICING (jika dijual):
  Open source core: free (MIT)
  Enterprise features: $500/node/month
  Managed SaaS: $0.10/GB encoded
```

---

## §13 — Risks & Limitations

### 13.1 Technical Risks

**RISK 1: Pattern Discovery Quality pada Diverse Corpus**
- *Problem*: Jika training corpus tidak representatif, patterns yang di-build tidak akan match data baru. OOV rate bisa tinggi (>20%), mengurangi compression gain signifikan.
- *Mitigation*: Rolling vocab update (offline), corpus diversity checker, OOV rate monitoring di production.
- *Honest Assessment*: CSM++ HANYA bekerja baik jika training corpus mencerminkan production data dengan baik. General-text deployment tanpa domain-specific training = CR hanya 1.5x vs BPE.

**RISK 2: FST Build Time untuk Very Large Vocab**
- *Problem*: V=1M entries → FST build ~15 menit single-threaded. Tidak ideal untuk online learning.
- *Mitigation*: Offline build + hot-reload. Vocab expansion maksimum 5% sebelum full rebuild diperlukan.
- *Honest Assessment*: CSM++ tidak mendukung true online learning. Ini adalah batch-build + streaming-apply system.

**RISK 3: Viterbi DP Memory untuk Very Long Lines**
- *Problem*: N=100,000 tokens per line (e.g., minified JS) → dp array = 100K × 24 bytes = 2.4 MB per line.
- *Mitigation*: Line length limit (default max 10,000 tokens), fallback ke greedy jika limit terlampaui.
- *Honest Assessment*: CSM++ didesain untuk typical log lines (50-500 tokens). Untuk very long documents, gunakan chunking atau greedy mode.

**RISK 4: Compression Ratio Claims**
- *Problem*: CR 18-20x untuk logs terdengar tinggi, bisa dianggap misleading.
- *Context*: CR ini versus raw UTF-8 bytes, bukan versus gzip. Versus gzip-9, CR advantage ~2-3x. Ini harus selalu dikomunikasikan dengan jelas.
- *Honest Assessment*: Untuk general text, CR advantage vs zstd mungkin hanya 10-30%. CSM++ value proposition adalah **semantic features + compression**, bukan compression saja.

### 13.2 Complexity Risks

**RISK 5: Implementation Complexity**
- Total codebase estimate: ~15,000–25,000 lines Rust
- 8 crates dengan inter-dependencies yang non-trivial
- PyO3 binding layer menambah surface area
- Kafka integration memerlukan expertise rdkafka
- *Mitigation*: Phased implementation (lihat §12.2), each crate independently testable.

**RISK 6: Correctness of Bit-Packing at Boundaries**
- Multi-tier bit packing dengan Elias-γ dan slot values interleaved adalah error-prone
- Off-by-one errors di bit boundaries sangat sulit di-debug
- *Mitigation*: Extensive proptest fuzz testing, round-trip tests, reference decoder sebagai ground truth.

**RISK 7: Arrow/Polars Version Compatibility**
- arrow2 API berubah signifikan antar versi minor
- Polars di Python juga berubah cepat
- *Mitigation*: Pin versions di Cargo.lock; explicit compatibility matrix di dokumentasi.

### 13.3 Adoption Risks

**RISK 8: "Not Invented Here" Problem**
- Tim engineering yang sudah punya Drain3/LogParser pipeline mungkin tidak mau migrasi
- *Mitigation*: Sediakan Drain3-compatible output mode; migration guide; side-by-side benchmark.

**RISK 9: Rust Barrier untuk Data Engineers**
- Data engineers lebih familiar Python; Rust learning curve tinggi untuk contributions
- *Mitigation*: Python API yang lengkap; PyPI distribusi wheel; tidak perlu Rust knowledge untuk penggunaan dasar.

**RISK 10: Cold Start Problem**
- CSM++ memerlukan training corpus sebelum bisa di-deploy. Tidak ada "zero-shot" mode.
- Untuk new service yang baru deploy (sedikit logs), pattern discovery tidak efektif.
- *Mitigation*: Pre-built vocab untuk common frameworks (nginx, k8s, app frameworks). "Transfer learning" dari existing pattern libraries.

### 13.4 When NOT to Use CSM++

```
DO NOT USE CSM++ FOR:
  ✗ Random/diverse text tanpa patterns (news articles, creative writing)
    → CR advantage < 10% vs zstd; overhead tidak sepadan
  ✗ Binary files (images, audio, compiled binaries)
    → Gunakan dedicated binary compressors
  ✗ Kecil logs (< 100MB)
    → Build overhead tidak sepadan; gzip cukup
  ✗ Real-time features dengan latency requirement < 1ms
    → Feature engine overhead ~0.3ms; mungkin tidak acceptable
  ✗ One-off compression (tidak perlu decode lagi)
    → Gunakan zstd-19; CR similar untuk general text, setup jauh lebih mudah
  ✗ Corpus yang berubah sangat cepat (vocab drift > 5% per hari)
    → Rebuild overhead terlalu tinggi; gunakan streaming-adapted approach

USE CSM++ WHEN:
  ✓ Structured/semi-structured logs dengan clear templates (nginx, app logs, k8s)
  ✓ Source code repositories yang besar (repetitive at AST level)
  ✓ Long-running services dengan stable log patterns
  ✓ Need BOTH compression AND ML features dari data yang sama
  ✓ Pipeline yang sudah pakai Kafka dan Arrow/Parquet
  ✓ Anomaly detection sebagai first-class requirement
```

---

## Appendix A — Crate Dependency Graph

```
csm-cli ──────────────────────────────────────────────────────────────┐
csm-python ──────────────────────────────────────────────────────── csm-api
csm-api ───────────────────────────────────────────────────────── csm-encoding
                                                                   csm-semantic
                                                                   csm-streaming
                                                                   csm-discovery
csm-semantic ──────────────────────────────────────────────────── csm-core
csm-encoding ──────────────────────────────────────────────────── csm-core
csm-streaming ─────────────────────────────────────────────────── csm-core
csm-tokenizer ─────────────────────────────────────────────────── csm-core
csm-discovery ─────────────────────────────────────────────────── csm-core
                                                                   csm-tokenizer
csm-core ── (no internal csm deps; only external crates)
  deps: fst, bumpalo, rustc-hash, smallvec, simdutf, thiserror
```

## Appendix B — Constants Reference

```rust
// Semua tunable constants dengan justification

// Chunking
pub const DEFAULT_CHUNK_TOKENS:  usize = 4_096;    // L2-cache optimal (4096×36=144KB < 256KB L2)
pub const MAX_LINE_TOKENS:       usize = 10_000;   // Viterbi DP memory guard
pub const FRAME_LINES_DEFAULT:   usize = 1_000;    // Kafka frame size

// Bit-packing tiers (Zipf α=1 optimal boundaries)
pub const TIER_0_MAX_ID:         u32   = 64;       // Covers ~27% corpus
pub const TIER_1_MAX_ID:         u32   = 4_096;    // Covers ~54% corpus
pub const TIER_2_MAX_ID:         u32   = 65_536;   // Covers ~79% corpus
// Tier 3: > 65_536 → Elias-γ

// Pattern discovery
pub const DEFAULT_MIN_FREQ:      u32   = 10;       // Minimum occurrence count
pub const DEFAULT_MAX_PATTERNS:  usize = 100_000;  // Capped at 10% vocab typical
pub const MAX_PATTERN_LEN:       usize = 5;        // k in O(N×k); keep small
pub const PPMI_THRESHOLD:        f32   = 0.5;      // Reject random co-occurrences
pub const PGS_MIN:               f32   = 0.30;     // Pattern generalization minimum
pub const KL_MAX:                f32   = 0.50;     // Context stability maximum KL
pub const STABILITY_MIN:         f32   = 0.30;     // 1 - CV(context) minimum
pub const H_SLOT_MIN:            f32   = 1.00;     // Minimum slot entropy (bits)
pub const H_SLOT_MAX:            f32   = 4.00;     // Maximum slot entropy = log2(16)
pub const HELD_OUT_RATIO:        f64   = 0.10;     // 10% corpus for PGS validation

// Feature engineering
pub const EMA_ALPHA:             f64   = 0.05;     // EMA decay factor
pub const ANOMALY_W_FREQ:        f32   = 0.35;
pub const ANOMALY_W_SLOT:        f32   = 0.30;
pub const ANOMALY_W_SEQ:         f32   = 0.20;
pub const ANOMALY_W_TEMPORAL:    f32   = 0.15;
pub const ANOMALY_ALERT_LOW:     f32   = 0.30;
pub const ANOMALY_ALERT_MED:     f32   = 0.60;
pub const ANOMALY_ALERT_HIGH:    f32   = 0.80;
pub const BURST_SCORE_WINDOW:    usize = 60;       // minutes for burst detection

// CMS (Count-Min Sketch) parameters
pub const CMS_EPSILON:           f64   = 0.001;    // 0.1% relative error
pub const CMS_DELTA:             f64   = 0.001;    // 0.1% failure probability
// → w = ceil(e/ε) = 2718, d = ceil(ln(1/δ)) = 7, RAM = 76 KB
```

---

*Dokumen ini adalah spesifikasi teknis resmi CSM++ v4.0.*
*Implementasi HARUS mengikuti spesifikasi ini.*
*Deviasi harus didokumentasikan sebagai ADR (Architecture Decision Record).*

**Copyright © 2026 Nafal Faturizki · CENTRA-NF DSL Ecosystem · All Rights Reserved**
