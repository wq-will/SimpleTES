use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use router_core::{route_qasm3_with_policy, CandidatePolicy, RouteOptions, RoutingTarget};

#[derive(Debug, Clone)]
struct SuiteCase {
    id: String,
    qasm3_path: String,
    topology_path: String,
    seed: Option<u64>,
}

#[derive(Debug, Clone)]
struct Topology {
    num_qubits: usize,
    edges: Vec<[usize; 2]>,
}

#[derive(Debug, Clone)]
struct CaseResult {
    id: String,
    ok: bool,
    swap_count: usize,
    depth: usize,
    twoq_count: usize,
    time_ms: u128,
    initial_mapping: Vec<usize>,
    output_circuit: String,
    error: Option<String>,
}

#[derive(Debug, Clone, Copy)]
struct TrialConfig {
    layout_trials: usize,
    routing_trials: usize,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let (suite_path, out_path) = parse_args()?;
    let trial_config = resolve_trial_config_from_env()?;
    let suite_text = fs::read_to_string(&suite_path)
        .map_err(|e| format!("failed to read suite {}: {e}", suite_path.display()))?;
    let suite_json = mini_json::parse(&suite_text)
        .map_err(|e| format!("failed to parse suite json {}: {e}", suite_path.display()))?;

    let suite_cases = parse_suite_cases(&suite_json)?;
    let suite_dir = suite_path.parent().unwrap_or_else(|| Path::new("."));

    let mut results = Vec::new();
    for case in &suite_cases {
        let start = Instant::now();
        let qasm_path = suite_dir.join(&case.qasm3_path);
        let topology_path = suite_dir.join(&case.topology_path);

        let result = match run_case_no_panic(case, &qasm_path, &topology_path, trial_config) {
            Ok((swap_count, depth, twoq_count, initial_mapping, output_circuit)) => CaseResult {
                id: case.id.clone(),
                ok: true,
                swap_count,
                depth,
                twoq_count,
                time_ms: start.elapsed().as_millis(),
                initial_mapping,
                output_circuit,
                error: None,
            },
            Err(err) => CaseResult {
                id: case.id.clone(),
                ok: false,
                swap_count: 0,
                depth: 0,
                twoq_count: 0,
                time_ms: start.elapsed().as_millis(),
                initial_mapping: Vec::new(),
                output_circuit: String::new(),
                error: Some(err),
            },
        };

        results.push(result);
    }

    let output_json = results_to_json(&results);
    fs::write(&out_path, output_json)
        .map_err(|e| format!("failed to write output {}: {e}", out_path.display()))?;
    Ok(())
}

fn parse_args() -> Result<(PathBuf, PathBuf), String> {
    let mut suite: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--suite" => {
                let Some(value) = args.next() else {
                    return Err("missing value for --suite".to_string());
                };
                suite = Some(PathBuf::from(value));
            }
            "--out" => {
                let Some(value) = args.next() else {
                    return Err("missing value for --out".to_string());
                };
                out = Some(PathBuf::from(value));
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                return Err(format!(
                    "unrecognized argument '{other}'. Use --help for usage."
                ));
            }
        }
    }

    let suite = suite.unwrap_or_else(default_suite_path);
    let Some(out) = out else {
        return Err("missing required --out <path>".to_string());
    };
    Ok((suite, out))
}

fn default_suite_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../python/benchmarks/sabre_suite.json")
}

fn print_help() {
    println!("router_cli [--suite <suite.json>] --out <results.json>");
    println!("default suite: {}", default_suite_path().display());
    println!(
        "optional env: QUBIT_ROUTING_LAYOUT_TRIALS, QUBIT_ROUTING_ROUTING_TRIALS (positive integers)"
    );
}

fn parse_positive_usize_env(name: &str) -> Result<Option<usize>, String> {
    let raw = match env::var(name) {
        Ok(value) => value,
        Err(env::VarError::NotPresent) => return Ok(None),
        Err(env::VarError::NotUnicode(_)) => {
            return Err(format!("environment variable '{name}' is not valid UTF-8"));
        }
    };
    let parsed = raw
        .parse::<usize>()
        .map_err(|_| format!("environment variable '{name}' must be a positive integer"))?;
    if parsed == 0 {
        return Err(format!(
            "environment variable '{name}' must be >= 1 (got 0)"
        ));
    }
    Ok(Some(parsed))
}

fn resolve_trial_config_from_env() -> Result<TrialConfig, String> {
    let defaults = RouteOptions::default();
    let layout_trials =
        parse_positive_usize_env("QUBIT_ROUTING_LAYOUT_TRIALS")?.unwrap_or(defaults.layout_trials);
    let routing_trials = parse_positive_usize_env("QUBIT_ROUTING_ROUTING_TRIALS")?
        .unwrap_or(defaults.routing_trials);
    Ok(TrialConfig {
        layout_trials,
        routing_trials,
    })
}

fn panic_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        return (*s).to_string();
    }
    if let Some(s) = payload.downcast_ref::<String>() {
        return s.clone();
    }
    "unknown panic payload".to_string()
}

fn run_case_no_panic(
    case: &SuiteCase,
    qasm_path: &Path,
    topology_path: &Path,
    trial_config: TrialConfig,
) -> Result<(usize, usize, usize, Vec<usize>, String), String> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let guarded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        process_case(case, qasm_path, topology_path, trial_config)
    }));
    std::panic::set_hook(hook);

    match guarded {
        Ok(result) => result,
        Err(payload) => Err(format!(
            "panic while processing case '{}': {}",
            case.id,
            panic_to_string(payload)
        )),
    }
}

fn process_case(
    case: &SuiteCase,
    qasm_path: &Path,
    topology_path: &Path,
    trial_config: TrialConfig,
) -> Result<(usize, usize, usize, Vec<usize>, String), String> {
    let qasm_text = fs::read_to_string(qasm_path)
        .map_err(|e| format!("failed to read qasm file {}: {e}", qasm_path.display()))?;
    let topology_text = fs::read_to_string(topology_path).map_err(|e| {
        format!(
            "failed to read topology file {}: {e}",
            topology_path.display()
        )
    })?;

    let topology_json = mini_json::parse(&topology_text)
        .map_err(|e| format!("invalid topology json {}: {e}", topology_path.display()))?;
    let topology = parse_topology(&topology_json)?;

    let target = RoutingTarget::new(topology.num_qubits, &topology.edges)
        .map_err(|e| format!("invalid topology for case '{}': {e}", case.id))?;

    let mut policy = CandidatePolicy::default();
    let options = RouteOptions {
        seed: case.seed.unwrap_or(7),
        layout_trials: trial_config.layout_trials,
        routing_trials: trial_config.routing_trials,
        ..RouteOptions::default()
    };

    let routed = route_qasm3_with_policy(&qasm_text, &target, &mut policy, options)
        .map_err(|e| e.to_string())?;
    Ok((
        routed.swap_count,
        routed.depth,
        routed.twoq_count,
        routed.initial_mapping,
        routed.output_circuit,
    ))
}

fn parse_suite_cases(json: &mini_json::JsonValue) -> Result<Vec<SuiteCase>, String> {
    let obj = json
        .as_object()
        .ok_or_else(|| "suite root must be a JSON object".to_string())?;
    let cases = obj
        .get("cases")
        .and_then(mini_json::JsonValue::as_array)
        .ok_or_else(|| "suite.cases must be an array".to_string())?;

    let mut out = Vec::new();
    for case in cases {
        let case_obj = case
            .as_object()
            .ok_or_else(|| "each suite case must be an object".to_string())?;

        let id = get_required_string(case_obj, "id")?;
        let qasm3_path = get_required_string(case_obj, "qasm3_path")?;
        let topology_path = get_required_string(case_obj, "topology_path")?;
        let seed = case_obj.get("seed").and_then(mini_json::JsonValue::as_u64);

        out.push(SuiteCase {
            id,
            qasm3_path,
            topology_path,
            seed,
        });
    }

    Ok(out)
}

fn parse_topology(json: &mini_json::JsonValue) -> Result<Topology, String> {
    let obj = json
        .as_object()
        .ok_or_else(|| "topology root must be a JSON object".to_string())?;

    let num_qubits =
        obj.get("num_qubits")
            .and_then(mini_json::JsonValue::as_u64)
            .ok_or_else(|| "topology.num_qubits must be an integer".to_string())? as usize;

    let edges_json = obj
        .get("edges")
        .and_then(mini_json::JsonValue::as_array)
        .ok_or_else(|| "topology.edges must be an array".to_string())?;

    let mut edges = Vec::new();
    for edge in edges_json {
        let arr = edge
            .as_array()
            .ok_or_else(|| "each edge must be a 2-element array".to_string())?;
        if arr.len() != 2 {
            return Err("each edge must be a 2-element array".to_string());
        }
        let a = arr[0]
            .as_u64()
            .ok_or_else(|| "edge endpoints must be integers".to_string())? as usize;
        let b = arr[1]
            .as_u64()
            .ok_or_else(|| "edge endpoints must be integers".to_string())? as usize;
        edges.push([a, b]);
    }

    Ok(Topology { num_qubits, edges })
}

fn get_required_string(
    obj: &BTreeMap<String, mini_json::JsonValue>,
    key: &str,
) -> Result<String, String> {
    obj.get(key)
        .and_then(mini_json::JsonValue::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("missing or invalid string field '{key}'"))
}

fn results_to_json(cases: &[CaseResult]) -> String {
    let mut case_values = Vec::new();
    for case in cases {
        let mut obj = BTreeMap::new();
        obj.insert("id".to_string(), mini_json::JsonValue::Str(case.id.clone()));
        obj.insert("ok".to_string(), mini_json::JsonValue::Bool(case.ok));
        obj.insert(
            "swap_count".to_string(),
            mini_json::JsonValue::Num(case.swap_count as f64),
        );
        obj.insert(
            "depth".to_string(),
            mini_json::JsonValue::Num(case.depth as f64),
        );
        obj.insert(
            "twoq_count".to_string(),
            mini_json::JsonValue::Num(case.twoq_count as f64),
        );
        obj.insert(
            "time_ms".to_string(),
            mini_json::JsonValue::Num(case.time_ms as f64),
        );
        obj.insert(
            "initial_mapping".to_string(),
            mini_json::JsonValue::Arr(
                case.initial_mapping
                    .iter()
                    .map(|&physical| mini_json::JsonValue::Num(physical as f64))
                    .collect(),
            ),
        );
        obj.insert(
            "output_circuit".to_string(),
            mini_json::JsonValue::Str(case.output_circuit.clone()),
        );
        obj.insert(
            "error".to_string(),
            match &case.error {
                Some(msg) => mini_json::JsonValue::Str(msg.clone()),
                None => mini_json::JsonValue::Null,
            },
        );
        case_values.push(mini_json::JsonValue::Obj(obj));
    }

    let mut root = BTreeMap::new();
    root.insert("cases".to_string(), mini_json::JsonValue::Arr(case_values));
    mini_json::stringify(&mini_json::JsonValue::Obj(root))
}

mod mini_json {
    use std::collections::BTreeMap;

    #[derive(Debug, Clone)]
    pub enum JsonValue {
        Null,
        Bool(bool),
        Num(f64),
        Str(String),
        Arr(Vec<JsonValue>),
        Obj(BTreeMap<String, JsonValue>),
    }

    impl JsonValue {
        pub fn as_object(&self) -> Option<&BTreeMap<String, JsonValue>> {
            match self {
                JsonValue::Obj(obj) => Some(obj),
                _ => None,
            }
        }

        pub fn as_array(&self) -> Option<&[JsonValue]> {
            match self {
                JsonValue::Arr(arr) => Some(arr),
                _ => None,
            }
        }

        pub fn as_str(&self) -> Option<&str> {
            match self {
                JsonValue::Str(s) => Some(s),
                _ => None,
            }
        }

        pub fn as_u64(&self) -> Option<u64> {
            match self {
                JsonValue::Num(n) if *n >= 0.0 && n.fract() == 0.0 => Some(*n as u64),
                _ => None,
            }
        }
    }

    pub fn parse(input: &str) -> Result<JsonValue, String> {
        let mut parser = Parser {
            bytes: input.as_bytes(),
            pos: 0,
        };
        let value = parser.parse_value()?;
        parser.skip_ws();
        if parser.pos != parser.bytes.len() {
            return Err("trailing characters after JSON value".to_string());
        }
        Ok(value)
    }

    pub fn stringify(value: &JsonValue) -> String {
        let mut out = String::new();
        write_value(&mut out, value);
        out
    }

    fn write_value(out: &mut String, value: &JsonValue) {
        match value {
            JsonValue::Null => out.push_str("null"),
            JsonValue::Bool(v) => out.push_str(if *v { "true" } else { "false" }),
            JsonValue::Num(n) => {
                if n.fract() == 0.0 {
                    out.push_str(&format!("{}", *n as i128));
                } else {
                    out.push_str(&n.to_string());
                }
            }
            JsonValue::Str(s) => {
                out.push('"');
                for ch in s.chars() {
                    match ch {
                        '"' => out.push_str("\\\""),
                        '\\' => out.push_str("\\\\"),
                        '\n' => out.push_str("\\n"),
                        '\r' => out.push_str("\\r"),
                        '\t' => out.push_str("\\t"),
                        c => out.push(c),
                    }
                }
                out.push('"');
            }
            JsonValue::Arr(arr) => {
                out.push('[');
                for (i, item) in arr.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_value(out, item);
                }
                out.push(']');
            }
            JsonValue::Obj(obj) => {
                out.push('{');
                for (i, (k, v)) in obj.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_value(out, &JsonValue::Str(k.clone()));
                    out.push(':');
                    write_value(out, v);
                }
                out.push('}');
            }
        }
    }

    struct Parser<'a> {
        bytes: &'a [u8],
        pos: usize,
    }

    impl<'a> Parser<'a> {
        fn parse_value(&mut self) -> Result<JsonValue, String> {
            self.skip_ws();
            let Some(ch) = self.peek() else {
                return Err("unexpected end of input".to_string());
            };
            match ch {
                b'n' => self.parse_literal("null", JsonValue::Null),
                b't' => self.parse_literal("true", JsonValue::Bool(true)),
                b'f' => self.parse_literal("false", JsonValue::Bool(false)),
                b'"' => self.parse_string().map(JsonValue::Str),
                b'[' => self.parse_array(),
                b'{' => self.parse_object(),
                b'-' | b'0'..=b'9' => self.parse_number(),
                _ => Err(format!("unexpected character '{}'", ch as char)),
            }
        }

        fn parse_literal(&mut self, lit: &str, value: JsonValue) -> Result<JsonValue, String> {
            if self.bytes.get(self.pos..self.pos + lit.len()) == Some(lit.as_bytes()) {
                self.pos += lit.len();
                Ok(value)
            } else {
                Err(format!("expected literal '{lit}'"))
            }
        }

        fn parse_string(&mut self) -> Result<String, String> {
            self.expect(b'"')?;
            let mut out = String::new();
            while let Some(ch) = self.next() {
                match ch {
                    b'"' => return Ok(out),
                    b'\\' => {
                        let Some(esc) = self.next() else {
                            return Err("unterminated escape sequence".to_string());
                        };
                        match esc {
                            b'"' => out.push('"'),
                            b'\\' => out.push('\\'),
                            b'/' => out.push('/'),
                            b'b' => out.push('\u{0008}'),
                            b'f' => out.push('\u{000C}'),
                            b'n' => out.push('\n'),
                            b'r' => out.push('\r'),
                            b't' => out.push('\t'),
                            _ => {
                                return Err(format!(
                                    "unsupported escape sequence '\\{}'",
                                    esc as char
                                ))
                            }
                        }
                    }
                    other => out.push(other as char),
                }
            }
            Err("unterminated string".to_string())
        }

        fn parse_array(&mut self) -> Result<JsonValue, String> {
            self.expect(b'[')?;
            let mut arr = Vec::new();
            loop {
                self.skip_ws();
                if self.peek() == Some(b']') {
                    self.pos += 1;
                    return Ok(JsonValue::Arr(arr));
                }
                arr.push(self.parse_value()?);
                self.skip_ws();
                match self.peek() {
                    Some(b',') => self.pos += 1,
                    Some(b']') => {
                        self.pos += 1;
                        return Ok(JsonValue::Arr(arr));
                    }
                    _ => return Err("expected ',' or ']' in array".to_string()),
                }
            }
        }

        fn parse_object(&mut self) -> Result<JsonValue, String> {
            self.expect(b'{')?;
            let mut obj = BTreeMap::new();
            loop {
                self.skip_ws();
                if self.peek() == Some(b'}') {
                    self.pos += 1;
                    return Ok(JsonValue::Obj(obj));
                }
                let key = self.parse_string()?;
                self.skip_ws();
                self.expect(b':')?;
                let value = self.parse_value()?;
                obj.insert(key, value);
                self.skip_ws();
                match self.peek() {
                    Some(b',') => self.pos += 1,
                    Some(b'}') => {
                        self.pos += 1;
                        return Ok(JsonValue::Obj(obj));
                    }
                    _ => return Err("expected ',' or '}' in object".to_string()),
                }
            }
        }

        fn parse_number(&mut self) -> Result<JsonValue, String> {
            let start = self.pos;
            if self.peek() == Some(b'-') {
                self.pos += 1;
            }
            self.consume_digits();
            if self.peek() == Some(b'.') {
                self.pos += 1;
                self.consume_digits();
            }
            if matches!(self.peek(), Some(b'e') | Some(b'E')) {
                self.pos += 1;
                if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                    self.pos += 1;
                }
                self.consume_digits();
            }
            let slice = std::str::from_utf8(&self.bytes[start..self.pos])
                .map_err(|_| "invalid utf8 in number".to_string())?;
            let num = slice
                .parse::<f64>()
                .map_err(|e| format!("invalid number '{slice}': {e}"))?;
            Ok(JsonValue::Num(num))
        }

        fn consume_digits(&mut self) {
            while let Some(ch) = self.peek() {
                if ch.is_ascii_digit() {
                    self.pos += 1;
                } else {
                    break;
                }
            }
        }

        fn expect(&mut self, expected: u8) -> Result<(), String> {
            match self.next() {
                Some(ch) if ch == expected => Ok(()),
                Some(ch) => Err(format!(
                    "expected '{}' but found '{}'",
                    expected as char, ch as char
                )),
                None => Err(format!(
                    "expected '{}' but reached end of input",
                    expected as char
                )),
            }
        }

        fn skip_ws(&mut self) {
            while let Some(ch) = self.peek() {
                if matches!(ch, b' ' | b'\n' | b'\r' | b'\t') {
                    self.pos += 1;
                } else {
                    break;
                }
            }
        }

        fn peek(&self) -> Option<u8> {
            self.bytes.get(self.pos).copied()
        }

        fn next(&mut self) -> Option<u8> {
            let ch = self.peek()?;
            self.pos += 1;
            Some(ch)
        }
    }
}
