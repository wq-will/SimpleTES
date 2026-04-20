// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::collections::HashMap;
use std::f64::consts::{E, PI, TAU};
use std::fmt::Write as _;

use num_bigint::BigUint;
use oq3_semantics::asg;
use oq3_semantics::symbols::{SymbolId, SymbolIdResult, SymbolTable, SymbolType};
use oq3_semantics::syntax_to_semantics::parse_source_string;
use oq3_semantics::types::{ArrayDims, Type};
use thiserror::Error;

use qiskit_circuit::bit::{ClassicalRegister, QuantumRegister};
use qiskit_circuit::circuit_data::{CircuitData, CircuitDataError};
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::operations::{
    Condition, ControlFlow, ControlFlowInstruction, ControlFlowView, Operation, Param,
    StandardGate, StandardInstruction,
};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::{Block, Clbit, Qubit, VarsMode};

#[derive(Debug, Error)]
pub enum RustImportError {
    #[error("qasm3 parse failed")]
    Parse,
    #[error("qasm3 import unsupported: {0}")]
    Unsupported(String),
    #[error("qasm3 import error: {0}")]
    Message(String),
}

impl From<CircuitDataError> for RustImportError {
    fn from(value: CircuitDataError) -> Self {
        Self::Message(value.to_string())
    }
}

#[derive(Default, Clone)]
struct WireMap {
    qregs: HashMap<SymbolId, Vec<Qubit>>,
    qubits: HashMap<SymbolId, Qubit>,
    cregs: HashMap<SymbolId, Vec<Clbit>>,
    creg_objs: HashMap<SymbolId, ClassicalRegister>,
    clbits: HashMap<SymbolId, Clbit>,
}

pub fn loads_to_dag(source: &str) -> Result<DAGCircuit, RustImportError> {
    let include_path: Vec<std::ffi::OsString> = Vec::new();
    let parsed = parse_source_string(source.to_owned(), None, Some(&include_path));
    if parsed.any_errors() {
        parsed.print_errors();
        return Err(RustImportError::Parse);
    }

    let symbols = parsed.symbol_table();
    let program = parsed.program();
    let mut dag = DAGCircuit::new();
    let mut wires = WireMap::default();

    import_statements(program.stmts(), &mut dag, &mut wires, symbols, true)?;
    Ok(dag)
}

pub fn dumps_from_dag(dag: &DAGCircuit) -> Result<String, RustImportError> {
    let data = CircuitData::from_dag_ref(dag).map_err(RustImportError::from)?;
    let exporter = crate::exporter::Exporter::new(
        vec!["stdgates.inc".to_string()],
        vec![],
        true,
        false,
        "  ".to_string(),
    );
    match exporter.dumps(&data, false) {
        Ok(qasm) => Ok(qasm),
        Err(err) => {
            if has_control_flow(dag) {
                dump_with_control_flow_fallback(dag)
            } else {
                Err(RustImportError::Message(err.to_string()))
            }
        }
    }
}

fn has_control_flow(dag: &DAGCircuit) -> bool {
    dag.topological_op_nodes(false)
        .any(|node| matches!(&dag[node], NodeType::Operation(inst) if dag.try_view_control_flow(inst).is_some()))
}

fn dump_with_control_flow_fallback(dag: &DAGCircuit) -> Result<String, RustImportError> {
    let mut out = String::new();
    out.push_str("OPENQASM 3.0;\n");
    out.push_str("include \"stdgates.inc\";\n");
    if dag.num_clbits() > 0 {
        let _ = writeln!(out, "bit[{}] c;", dag.num_clbits());
    }
    if dag.num_qubits() > 0 {
        let _ = writeln!(out, "qubit[{}] q;", dag.num_qubits());
    }
    dump_ops_with_indent(dag, 0, &mut out)?;
    Ok(out)
}

fn dump_ops_with_indent(
    dag: &DAGCircuit,
    indent: usize,
    out: &mut String,
) -> Result<(), RustImportError> {
    for node in dag.topological_op_nodes(false) {
        let NodeType::Operation(inst) = &dag[node] else {
            continue;
        };
        if let Some(cf) = dag.try_view_control_flow(inst) {
            match cf {
                ControlFlowView::IfElse {
                    condition,
                    true_body,
                    false_body,
                } => {
                    write_indent(out, indent);
                    let cond = format_condition(condition, dag)?;
                    let _ = writeln!(out, "if ({cond}) {{");
                    dump_ops_with_indent(true_body, indent + 1, out)?;
                    write_indent(out, indent);
                    out.push('}');
                    if let Some(body) = false_body {
                        out.push_str(" else {\n");
                        dump_ops_with_indent(body, indent + 1, out)?;
                        write_indent(out, indent);
                        out.push('}');
                    }
                    out.push('\n');
                }
                _ => {
                    return Err(RustImportError::Unsupported(
                        "control-flow fallback only supports if/else".to_string(),
                    ));
                }
            }
            continue;
        }
        dump_non_cf_op(dag, inst, indent, out)?;
    }
    Ok(())
}

fn dump_non_cf_op(
    dag: &DAGCircuit,
    inst: &qiskit_circuit::packed_instruction::PackedInstruction,
    indent: usize,
    out: &mut String,
) -> Result<(), RustImportError> {
    write_indent(out, indent);
    if let Some(gate) = inst.op.try_standard_gate() {
        if gate.num_params() != 0 {
            return Err(RustImportError::Unsupported(format!(
                "fallback export does not support parameterized gate '{}'",
                gate.name()
            )));
        }
        let qargs = dag
            .get_qargs(inst.qubits)
            .iter()
            .map(|q| format!("q[{}]", q.index()))
            .collect::<Vec<_>>();
        let _ = writeln!(out, "{} {};", gate.name(), qargs.join(", "));
        return Ok(());
    }

    if let Some(instr) = inst.op.try_standard_instruction() {
        match instr {
            StandardInstruction::Measure => {
                let qargs = dag.get_qargs(inst.qubits);
                let cargs = dag.get_cargs(inst.clbits);
                if qargs.len() != cargs.len() {
                    return Err(RustImportError::Message(
                        "measure instruction has mismatched qargs/cargs".to_string(),
                    ));
                }
                for (q, c) in qargs.iter().zip(cargs.iter()) {
                    write_indent(out, indent);
                    let _ = writeln!(out, "c[{}] = measure q[{}];", c.index(), q.index());
                }
                return Ok(());
            }
            StandardInstruction::Reset => {
                let qargs = dag.get_qargs(inst.qubits);
                if qargs.len() != 1 {
                    return Err(RustImportError::Message(
                        "reset instruction expected one qubit".to_string(),
                    ));
                }
                let _ = writeln!(out, "reset q[{}];", qargs[0].index());
                return Ok(());
            }
            StandardInstruction::Barrier(_) => {
                let qargs = dag
                    .get_qargs(inst.qubits)
                    .iter()
                    .map(|q| format!("q[{}]", q.index()))
                    .collect::<Vec<_>>();
                let _ = writeln!(out, "barrier {};", qargs.join(", "));
                return Ok(());
            }
            _ => {
                return Err(RustImportError::Unsupported(format!(
                    "fallback export does not support instruction '{}'",
                    instr.name()
                )));
            }
        }
    }

    Err(RustImportError::Unsupported(format!(
        "fallback export does not support operation '{}'",
        inst.op.name()
    )))
}

fn format_condition(condition: &Condition, dag: &DAGCircuit) -> Result<String, RustImportError> {
    match condition {
        Condition::Bit(bit, expected) => {
            let idx = dag.clbits().find(bit).ok_or_else(|| {
                RustImportError::Message(
                    "failed to resolve clbit index for if-condition".to_string(),
                )
            })?;
            if *expected {
                Ok(format!("c[{}]", idx.index()))
            } else {
                Ok(format!("!c[{}]", idx.index()))
            }
        }
        Condition::Register(_reg, value) => Ok(format!("c == {value}")),
        Condition::Expr(_) => Err(RustImportError::Unsupported(
            "expression-based conditions are not supported in fallback export".to_string(),
        )),
    }
}

fn write_indent(out: &mut String, indent: usize) {
    for _ in 0..indent {
        out.push_str("  ");
    }
}

fn import_statements(
    statements: &[asg::Stmt],
    dag: &mut DAGCircuit,
    wires: &mut WireMap,
    symbols: &SymbolTable,
    allow_decls: bool,
) -> Result<(), RustImportError> {
    for stmt in statements {
        match stmt {
            asg::Stmt::DeclareQuantum(decl) => {
                if !allow_decls {
                    return Err(RustImportError::Unsupported(
                        "declarations inside control-flow blocks are not supported".to_string(),
                    ));
                }
                declare_quantum(dag, wires, symbols, decl)?;
            }
            asg::Stmt::DeclareClassical(decl) => {
                if !allow_decls {
                    return Err(RustImportError::Unsupported(
                        "declarations inside control-flow blocks are not supported".to_string(),
                    ));
                }
                declare_classical(dag, wires, symbols, decl)?;
            }
            asg::Stmt::GateCall(call) => {
                import_gate_call(dag, wires, symbols, call)?;
            }
            asg::Stmt::Barrier(barrier) => {
                import_barrier(dag, wires, symbols, barrier)?;
            }
            asg::Stmt::Assignment(assignment) => {
                import_assignment(dag, wires, symbols, assignment)?;
            }
            asg::Stmt::Reset(reset) => {
                import_reset(dag, wires, symbols, reset)?;
            }
            asg::Stmt::If(if_stmt) => {
                import_if(dag, wires, symbols, if_stmt)?;
            }
            asg::Stmt::Include(_)
            | asg::Stmt::GateDefinition(_)
            | asg::Stmt::Pragma(_)
            | asg::Stmt::NullStmt
            | asg::Stmt::AnnotatedStmt(_) => {}
            asg::Stmt::Alias(_)
            | asg::Stmt::Block(_)
            | asg::Stmt::Box
            | asg::Stmt::Break
            | asg::Stmt::Cal
            | asg::Stmt::Continue
            | asg::Stmt::DeclareHardwareQubit(_)
            | asg::Stmt::DefCal
            | asg::Stmt::DefStmt(_)
            | asg::Stmt::Delay(_)
            | asg::Stmt::End
            | asg::Stmt::ExprStmt(_)
            | asg::Stmt::Extern
            | asg::Stmt::ForStmt(_)
            | asg::Stmt::GPhaseCall(_)
            | asg::Stmt::InputDeclaration(_)
            | asg::Stmt::ModifiedGPhaseCall(_)
            | asg::Stmt::OldStyleDeclaration
            | asg::Stmt::OutputDeclaration(_)
            | asg::Stmt::SwitchCaseStmt(_)
            | asg::Stmt::While(_) => {
                return Err(RustImportError::Unsupported(format!(
                    "statement not supported by Rust importer: {stmt:?}"
                )));
            }
        }
    }
    Ok(())
}

fn declare_quantum(
    dag: &mut DAGCircuit,
    wires: &mut WireMap,
    symbols: &SymbolTable,
    decl: &asg::DeclareQuantum,
) -> Result<(), RustImportError> {
    let name_id = resolve_symbol(decl.name())?;
    let name = symbols[&name_id].name().to_owned();

    match symbols[&name_id].symbol_type() {
        Type::Qubit => {
            let start = dag.num_qubits() as u32;
            let reg = QuantumRegister::new_owning(name, 1);
            dag.add_qreg(reg)
                .map_err(|e| RustImportError::Message(e.to_string()))?;
            let qubit = Qubit(start);
            wires.qubits.insert(name_id.clone(), qubit);
            wires.qregs.insert(name_id, vec![qubit]);
        }
        Type::QubitArray(ArrayDims::D1(size)) => {
            let start = dag.num_qubits() as u32;
            let reg = QuantumRegister::new_owning(name, *size as u32);
            dag.add_qreg(reg)
                .map_err(|e| RustImportError::Message(e.to_string()))?;
            let bits = (0..*size)
                .map(|offset| Qubit(start + offset as u32))
                .collect::<Vec<_>>();
            wires.qregs.insert(name_id, bits);
        }
        ty => {
            return Err(RustImportError::Unsupported(format!(
                "unsupported quantum declaration type: {ty:?}"
            )));
        }
    }

    Ok(())
}

fn declare_classical(
    dag: &mut DAGCircuit,
    wires: &mut WireMap,
    symbols: &SymbolTable,
    decl: &asg::DeclareClassical,
) -> Result<(), RustImportError> {
    if decl.initializer().is_some() {
        return Err(RustImportError::Unsupported(
            "initialized classical declarations are not supported".to_string(),
        ));
    }

    let name_id = resolve_symbol(decl.name())?;
    let name = symbols[&name_id].name().to_owned();

    match symbols[&name_id].symbol_type() {
        Type::Bit(_) => {
            let start = dag.num_clbits() as u32;
            let reg = ClassicalRegister::new_owning(name, 1);
            dag.add_creg(reg.clone())
                .map_err(|e| RustImportError::Message(e.to_string()))?;
            let clbit = Clbit(start);
            wires.clbits.insert(name_id.clone(), clbit);
            wires.creg_objs.insert(name_id.clone(), reg);
            wires.cregs.insert(name_id, vec![clbit]);
        }
        Type::BitArray(ArrayDims::D1(size), _) => {
            let start = dag.num_clbits() as u32;
            let reg = ClassicalRegister::new_owning(name, *size as u32);
            dag.add_creg(reg.clone())
                .map_err(|e| RustImportError::Message(e.to_string()))?;
            let bits = (0..*size)
                .map(|offset| Clbit(start + offset as u32))
                .collect::<Vec<_>>();
            wires.creg_objs.insert(name_id.clone(), reg);
            wires.cregs.insert(name_id, bits);
        }
        ty => {
            return Err(RustImportError::Unsupported(format!(
                "unsupported classical declaration type: {ty:?}"
            )));
        }
    }

    Ok(())
}

fn import_gate_call(
    dag: &mut DAGCircuit,
    wires: &WireMap,
    symbols: &SymbolTable,
    call: &asg::GateCall,
) -> Result<(), RustImportError> {
    if !call.modifiers().is_empty() {
        return Err(RustImportError::Unsupported(
            "gate modifiers are not supported in Rust importer".to_string(),
        ));
    }

    let gate_id = resolve_symbol(call.name())?;
    let gate_name = symbols[&gate_id].name();
    let gate = parse_standard_gate(gate_name).ok_or_else(|| {
        RustImportError::Unsupported(format!("unsupported gate in Rust importer: '{gate_name}'"))
    })?;

    let mut params = Vec::<Param>::new();
    if let Some(raw_params) = call.params() {
        params.reserve(raw_params.len());
        for param in raw_params {
            params.push(Param::Float(eval_float_expr(param, symbols)?));
        }
    }

    if params.len() != gate.num_params() as usize {
        return Err(RustImportError::Message(format!(
            "incorrect parameter count for gate '{gate_name}': expected {}, got {}",
            gate.num_params(),
            params.len()
        )));
    }

    let qarg_lists = call
        .qubits()
        .iter()
        .map(|expr| eval_qubits(expr, wires, symbols))
        .collect::<Result<Vec<_>, _>>()?;

    for qargs in broadcast_lists(&qarg_lists)? {
        if qargs.len() != gate.num_qubits() as usize {
            return Err(RustImportError::Message(format!(
                "incorrect qubit count for gate '{gate_name}': expected {}, got {}",
                gate.num_qubits(),
                qargs.len()
            )));
        }

        let params = if params.is_empty() {
            None
        } else {
            Some(Parameters::Params(params.clone().into()))
        };

        dag.apply_operation_back(
            PackedOperation::from_standard_gate(gate),
            &qargs,
            &[],
            params,
            None,
            #[cfg(feature = "cache_pygates")]
            None,
        )
        .map_err(|e| RustImportError::Message(e.to_string()))?;
    }

    Ok(())
}

fn import_barrier(
    dag: &mut DAGCircuit,
    wires: &WireMap,
    symbols: &SymbolTable,
    barrier: &asg::Barrier,
) -> Result<(), RustImportError> {
    let qargs = if let Some(exprs) = barrier.qubits() {
        let mut out = Vec::<Qubit>::new();
        for expr in exprs {
            for q in eval_qubits(expr, wires, symbols)? {
                if !out.contains(&q) {
                    out.push(q);
                }
            }
        }
        out
    } else {
        (0..dag.num_qubits()).map(|idx| Qubit::new(idx)).collect()
    };

    dag.apply_operation_back(
        PackedOperation::from_standard_instruction(
            StandardInstruction::Barrier(qargs.len() as u32),
        ),
        &qargs,
        &[],
        None,
        None,
        #[cfg(feature = "cache_pygates")]
        None,
    )
    .map_err(|e| RustImportError::Message(e.to_string()))?;

    Ok(())
}

fn import_assignment(
    dag: &mut DAGCircuit,
    wires: &WireMap,
    symbols: &SymbolTable,
    assignment: &asg::Assignment,
) -> Result<(), RustImportError> {
    let asg::Expr::MeasureExpression(measure) = assignment.rvalue().expression() else {
        return Err(RustImportError::Unsupported(
            "only measurement assignments are supported".to_string(),
        ));
    };

    let qubits = eval_qubits(measure.operand(), wires, symbols)?;
    let clbits = eval_lvalue_clbits(assignment.lvalue(), wires, symbols)?;
    let pairs = broadcast_measure_pairs(&qubits, &clbits)?;

    for (qubit, clbit) in pairs {
        dag.apply_operation_back(
            PackedOperation::from_standard_instruction(StandardInstruction::Measure),
            &[qubit],
            &[clbit],
            None,
            None,
            #[cfg(feature = "cache_pygates")]
            None,
        )
        .map_err(|e| RustImportError::Message(e.to_string()))?;
    }

    Ok(())
}

fn import_reset(
    dag: &mut DAGCircuit,
    wires: &WireMap,
    symbols: &SymbolTable,
    reset: &asg::Reset,
) -> Result<(), RustImportError> {
    for qubit in eval_qubits(reset.gate_operand(), wires, symbols)? {
        dag.apply_operation_back(
            PackedOperation::from_standard_instruction(StandardInstruction::Reset),
            &[qubit],
            &[],
            None,
            None,
            #[cfg(feature = "cache_pygates")]
            None,
        )
        .map_err(|e| RustImportError::Message(e.to_string()))?;
    }

    Ok(())
}

fn import_if(
    dag: &mut DAGCircuit,
    wires: &mut WireMap,
    symbols: &SymbolTable,
    if_stmt: &asg::If,
) -> Result<(), RustImportError> {
    let condition = eval_condition(if_stmt.condition(), dag, wires, symbols)?;
    let mut blocks = Vec::<DAGCircuit>::new();

    let mut true_block = dag
        .copy_empty_like(VarsMode::Drop, qiskit_circuit::BlocksMode::Drop)
        .map_err(|e| RustImportError::Message(e.to_string()))?;
    import_statements(
        if_stmt.then_branch().statements(),
        &mut true_block,
        wires,
        symbols,
        false,
    )?;
    blocks.push(true_block);

    if let Some(false_branch) = if_stmt.else_branch() {
        let mut false_block = dag
            .copy_empty_like(VarsMode::Drop, qiskit_circuit::BlocksMode::Drop)
            .map_err(|e| RustImportError::Message(e.to_string()))?;
        import_statements(
            false_branch.statements(),
            &mut false_block,
            wires,
            symbols,
            false,
        )?;
        blocks.push(false_block);
    }

    let block_ids = blocks
        .into_iter()
        .map(|block| dag.add_block(block))
        .collect::<Vec<Block>>();

    let qargs = (0..dag.num_qubits()).map(Qubit::new).collect::<Vec<_>>();
    let cargs = condition_clbits(&condition, dag, wires)?;

    let op = ControlFlowInstruction {
        control_flow: ControlFlow::IfElse { condition },
        num_qubits: qargs.len() as u32,
        num_clbits: cargs.len() as u32,
    };

    dag.apply_operation_back(
        PackedOperation::from_control_flow(Box::new(op)),
        &qargs,
        &cargs,
        Some(Parameters::Blocks(block_ids)),
        None,
        #[cfg(feature = "cache_pygates")]
        None,
    )
    .map_err(|e| RustImportError::Message(e.to_string()))?;

    Ok(())
}

fn condition_clbits(
    condition: &Condition,
    dag: &DAGCircuit,
    wires: &WireMap,
) -> Result<Vec<Clbit>, RustImportError> {
    match condition {
        Condition::Bit(bit, _) => dag
            .clbits()
            .find(bit)
            .map(|mapped| vec![mapped])
            .ok_or_else(|| {
                RustImportError::Message(
                    "failed to resolve classical bit for if-condition".to_string(),
                )
            }),
        Condition::Register(creg, _) => {
            let bits = wires
                .creg_objs
                .iter()
                .find_map(|(sid, reg)| (reg == creg).then(|| sid))
                .and_then(|sid| wires.cregs.get(sid))
                .cloned()
                .ok_or_else(|| {
                    RustImportError::Message(
                        "failed to resolve condition register bits".to_string(),
                    )
                })?;
            Ok(bits)
        }
        Condition::Expr(_) => Err(RustImportError::Unsupported(
            "expression conditions are not supported in Rust importer".to_string(),
        )),
    }
}

fn eval_condition(
    expr: &asg::TExpr,
    dag: &DAGCircuit,
    wires: &WireMap,
    symbols: &SymbolTable,
) -> Result<Condition, RustImportError> {
    match expr.expression() {
        asg::Expr::Identifier(id) => {
            let id = resolve_symbol(id)?;
            if let Some(clbit) = wires.clbits.get(&id).copied() {
                let shareable = dag.clbits().get(clbit).cloned().ok_or_else(|| {
                    RustImportError::Message(format!(
                        "failed to resolve classical bit {} for condition",
                        clbit.index()
                    ))
                })?;
                return Ok(Condition::Bit(shareable, true));
            }
            if let Some(creg) = wires.creg_objs.get(&id).cloned() {
                return Ok(Condition::Register(creg, BigUint::from(1u8)));
            }
            Err(RustImportError::Unsupported(
                "unsupported condition identifier".to_string(),
            ))
        }
        asg::Expr::IndexedIdentifier(idx) => {
            let bits = eval_indexed_clbits(idx, wires, symbols)?;
            if bits.len() != 1 {
                return Err(RustImportError::Unsupported(
                    "condition indexed identifier must resolve to one bit".to_string(),
                ));
            }
            let shareable = dag
                .clbits()
                .get(bits[0])
                .cloned()
                .ok_or_else(|| RustImportError::Message("missing condition bit".to_string()))?;
            Ok(Condition::Bit(shareable, true))
        }
        asg::Expr::BinaryExpr(binary) => eval_binary_condition(binary, dag, wires, symbols),
        _ => Err(RustImportError::Unsupported(
            "unsupported condition expression".to_string(),
        )),
    }
}

fn eval_binary_condition(
    binary: &asg::BinaryExpr,
    dag: &DAGCircuit,
    wires: &WireMap,
    symbols: &SymbolTable,
) -> Result<Condition, RustImportError> {
    let asg::BinaryOp::CmpOp(cmp) = binary.op() else {
        return Err(RustImportError::Unsupported(
            "if condition must be a comparison or bit/register".to_string(),
        ));
    };

    if let Some((bit, value)) =
        eval_condition_bit_compare(binary.left(), binary.right(), wires, symbols)?
    {
        let expected = match cmp {
            asg::CmpOp::Eq => value,
            asg::CmpOp::Neq => !value,
        };
        let shareable = dag
            .clbits()
            .get(bit)
            .cloned()
            .ok_or_else(|| RustImportError::Message("missing condition bit".to_string()))?;
        return Ok(Condition::Bit(shareable, expected));
    }

    if let Some((reg, value)) =
        eval_condition_reg_compare(binary.left(), binary.right(), wires, symbols)?
    {
        match cmp {
            asg::CmpOp::Eq => return Ok(Condition::Register(reg, value)),
            asg::CmpOp::Neq => {
                return Err(RustImportError::Unsupported(
                    "register != literal conditions are not supported".to_string(),
                ));
            }
        }
    }

    Err(RustImportError::Unsupported(
        "unsupported if-condition comparison".to_string(),
    ))
}

fn eval_condition_bit_compare(
    left: &asg::TExpr,
    right: &asg::TExpr,
    wires: &WireMap,
    symbols: &SymbolTable,
) -> Result<Option<(Clbit, bool)>, RustImportError> {
    let left_bit = eval_clbit_scalar(left, wires, symbols)?;
    let right_bit = eval_clbit_scalar(right, wires, symbols)?;

    if let (Some(bit), Some(value)) = (left_bit, eval_bool_literal(right, symbols)?) {
        return Ok(Some((bit, value)));
    }
    if let (Some(bit), Some(value)) = (right_bit, eval_bool_literal(left, symbols)?) {
        return Ok(Some((bit, value)));
    }

    Ok(None)
}

fn eval_condition_reg_compare(
    left: &asg::TExpr,
    right: &asg::TExpr,
    wires: &WireMap,
    symbols: &SymbolTable,
) -> Result<Option<(ClassicalRegister, BigUint)>, RustImportError> {
    if let Some(reg) = eval_creg_scalar(left, wires, symbols)? {
        if let Some(value) = eval_nonnegative_int(right, symbols)? {
            return Ok(Some((reg, value)));
        }
    }
    if let Some(reg) = eval_creg_scalar(right, wires, symbols)? {
        if let Some(value) = eval_nonnegative_int(left, symbols)? {
            return Ok(Some((reg, value)));
        }
    }
    Ok(None)
}

fn eval_creg_scalar(
    expr: &asg::TExpr,
    wires: &WireMap,
    _symbols: &SymbolTable,
) -> Result<Option<ClassicalRegister>, RustImportError> {
    match expr.expression() {
        asg::Expr::Identifier(id) => {
            let id = resolve_symbol(id)?;
            Ok(wires.creg_objs.get(&id).cloned())
        }
        _ => Ok(None),
    }
}

fn eval_clbit_scalar(
    expr: &asg::TExpr,
    wires: &WireMap,
    symbols: &SymbolTable,
) -> Result<Option<Clbit>, RustImportError> {
    match expr.expression() {
        asg::Expr::Identifier(id) => {
            let id = resolve_symbol(id)?;
            Ok(wires.clbits.get(&id).copied())
        }
        asg::Expr::IndexedIdentifier(idx) => {
            let bits = eval_indexed_clbits(idx, wires, symbols)?;
            Ok((bits.len() == 1).then_some(bits[0]))
        }
        _ => Ok(None),
    }
}

fn eval_bool_literal(
    expr: &asg::TExpr,
    symbols: &SymbolTable,
) -> Result<Option<bool>, RustImportError> {
    match expr.expression() {
        asg::Expr::Literal(asg::Literal::Bool(v)) => Ok(Some(*v.value())),
        _ => {
            if let Some(value) = eval_nonnegative_int(expr, symbols)? {
                if value == BigUint::from(0u8) {
                    return Ok(Some(false));
                }
                if value == BigUint::from(1u8) {
                    return Ok(Some(true));
                }
            }
            Ok(None)
        }
    }
}

fn eval_nonnegative_int(
    expr: &asg::TExpr,
    symbols: &SymbolTable,
) -> Result<Option<BigUint>, RustImportError> {
    let value = eval_int_expr(expr, symbols)?;
    if value < 0 {
        return Ok(None);
    }
    Ok(Some(BigUint::from(value as u128)))
}

fn eval_lvalue_clbits(
    lvalue: &asg::LValue,
    wires: &WireMap,
    symbols: &SymbolTable,
) -> Result<Vec<Clbit>, RustImportError> {
    match lvalue {
        asg::LValue::Identifier(id) => {
            let id = resolve_symbol(id)?;
            if let Some(bit) = wires.clbits.get(&id).copied() {
                return Ok(vec![bit]);
            }
            if let Some(bits) = wires.cregs.get(&id) {
                return Ok(bits.clone());
            }
            Err(RustImportError::Message(format!(
                "unknown classical lvalue symbol: {}",
                symbols[&id].name()
            )))
        }
        asg::LValue::IndexedIdentifier(idx) => eval_indexed_clbits(idx, wires, symbols),
    }
}

fn eval_qubits(
    expr: &asg::TExpr,
    wires: &WireMap,
    symbols: &SymbolTable,
) -> Result<Vec<Qubit>, RustImportError> {
    match expr.expression() {
        asg::Expr::GateOperand(op) => eval_gate_operand_qubits(op, wires, symbols),
        asg::Expr::Identifier(id) => {
            let id = resolve_symbol(id)?;
            if let Some(bit) = wires.qubits.get(&id).copied() {
                return Ok(vec![bit]);
            }
            if let Some(bits) = wires.qregs.get(&id) {
                return Ok(bits.clone());
            }
            Err(RustImportError::Message(format!(
                "unknown quantum symbol: {}",
                symbols[&id].name()
            )))
        }
        asg::Expr::IndexedIdentifier(idx) => eval_indexed_qubits(idx, wires, symbols),
        other => Err(RustImportError::Unsupported(format!(
            "unsupported quantum operand expression: {other:?}"
        ))),
    }
}

fn eval_gate_operand_qubits(
    op: &asg::GateOperand,
    wires: &WireMap,
    symbols: &SymbolTable,
) -> Result<Vec<Qubit>, RustImportError> {
    match op {
        asg::GateOperand::Identifier(id) => {
            let id = resolve_symbol(id)?;
            if let Some(bit) = wires.qubits.get(&id).copied() {
                return Ok(vec![bit]);
            }
            if let Some(bits) = wires.qregs.get(&id) {
                return Ok(bits.clone());
            }
            Err(RustImportError::Message(format!(
                "unknown quantum operand: {}",
                symbols[&id].name()
            )))
        }
        asg::GateOperand::IndexedIdentifier(idx) => eval_indexed_qubits(idx, wires, symbols),
        asg::GateOperand::HardwareQubit(_) => Err(RustImportError::Unsupported(
            "hardware qubit operands are not supported".to_string(),
        )),
    }
}

fn eval_indexed_qubits(
    indexed: &asg::IndexedIdentifier,
    wires: &WireMap,
    symbols: &SymbolTable,
) -> Result<Vec<Qubit>, RustImportError> {
    let id = resolve_symbol(indexed.identifier())?;
    let register = wires.qregs.get(&id).ok_or_else(|| {
        RustImportError::Message(format!(
            "indexed quantum operand '{}' is not a quantum register",
            symbols[&id].name()
        ))
    })?;
    let indices = eval_indices(indexed.indexes(), symbols)?;
    materialize_indices(register, &indices, symbols[&id].name(), "quantum")
}

fn eval_indexed_clbits(
    indexed: &asg::IndexedIdentifier,
    wires: &WireMap,
    symbols: &SymbolTable,
) -> Result<Vec<Clbit>, RustImportError> {
    let id = resolve_symbol(indexed.identifier())?;
    let register = wires.cregs.get(&id).ok_or_else(|| {
        RustImportError::Message(format!(
            "indexed classical operand '{}' is not a classical register",
            symbols[&id].name()
        ))
    })?;
    let indices = eval_indices(indexed.indexes(), symbols)?;
    materialize_indices(register, &indices, symbols[&id].name(), "classical")
}

fn materialize_indices<T: Copy>(
    bits: &[T],
    indices: &[usize],
    name: &str,
    kind: &str,
) -> Result<Vec<T>, RustImportError> {
    let mut out = Vec::with_capacity(indices.len());
    for &idx in indices {
        if idx >= bits.len() {
            return Err(RustImportError::Message(format!(
                "{kind} index {idx} out of range for register '{name}' of size {}",
                bits.len()
            )));
        }
        out.push(bits[idx]);
    }
    Ok(out)
}

fn eval_indices(
    indexes: &[asg::IndexOperator],
    symbols: &SymbolTable,
) -> Result<Vec<usize>, RustImportError> {
    if indexes.len() != 1 {
        return Err(RustImportError::Unsupported(
            "only one-dimensional indexing is supported".to_string(),
        ));
    }

    match &indexes[0] {
        asg::IndexOperator::ExpressionList(list) => {
            if list.expressions.len() == 1 {
                if let asg::Expr::RangeExpression(range) = list.expressions[0].expression() {
                    return eval_range(range, symbols);
                }
            }
            list.expressions
                .iter()
                .map(|expr| eval_index_value(expr, symbols))
                .collect()
        }
        asg::IndexOperator::SetExpression(set) => set
            .expressions()
            .iter()
            .map(|expr| eval_index_value(expr, symbols))
            .collect(),
    }
}

fn eval_range(
    range: &asg::RangeExpression,
    symbols: &SymbolTable,
) -> Result<Vec<usize>, RustImportError> {
    let start = eval_int_expr(range.start(), symbols)?;
    let stop = eval_int_expr(range.stop(), symbols)?;
    let step = if let Some(step_expr) = range.step() {
        eval_int_expr(step_expr, symbols)?
    } else if stop >= start {
        1
    } else {
        -1
    };
    if step == 0 {
        return Err(RustImportError::Message(
            "range step cannot be zero".to_string(),
        ));
    }

    let mut out = Vec::new();
    let mut current = start;
    if step > 0 {
        while current <= stop {
            if current < 0 {
                return Err(RustImportError::Message(
                    "range contains negative index".to_string(),
                ));
            }
            out.push(current as usize);
            current += step;
        }
    } else {
        while current >= stop {
            if current < 0 {
                return Err(RustImportError::Message(
                    "range contains negative index".to_string(),
                ));
            }
            out.push(current as usize);
            current += step;
        }
    }
    Ok(out)
}

fn eval_index_value(expr: &asg::TExpr, symbols: &SymbolTable) -> Result<usize, RustImportError> {
    let value = eval_int_expr(expr, symbols)?;
    if value < 0 {
        return Err(RustImportError::Message(
            "index cannot be negative".to_string(),
        ));
    }
    Ok(value as usize)
}

fn eval_int_expr(expr: &asg::TExpr, symbols: &SymbolTable) -> Result<i128, RustImportError> {
    match expr.expression() {
        asg::Expr::Literal(asg::Literal::Int(value)) => {
            let mag = *value.value() as i128;
            if *value.sign() {
                Ok(mag)
            } else {
                Ok(-mag)
            }
        }
        asg::Expr::Literal(asg::Literal::Bool(value)) => Ok(if *value.value() { 1 } else { 0 }),
        asg::Expr::Identifier(id) => {
            let id = resolve_symbol(id)?;
            match symbols[&id].name() {
                "pi" | "π" => Ok(PI as i128),
                "tau" | "τ" => Ok(TAU as i128),
                "euler" | "ℇ" => Ok(E as i128),
                other => Err(RustImportError::Unsupported(format!(
                    "identifier '{other}' is not a supported integer constant"
                ))),
            }
        }
        asg::Expr::UnaryExpr(unary) => {
            let value = eval_int_expr(unary.operand(), symbols)?;
            match unary.op() {
                asg::UnaryOp::Minus => Ok(-value),
                asg::UnaryOp::Not => Ok((value == 0) as i128),
                asg::UnaryOp::BitNot => Ok(!value),
            }
        }
        asg::Expr::BinaryExpr(binary) => {
            let left = eval_int_expr(binary.left(), symbols)?;
            let right = eval_int_expr(binary.right(), symbols)?;
            match binary.op() {
                asg::BinaryOp::ArithOp(asg::ArithOp::Add) => Ok(left + right),
                asg::BinaryOp::ArithOp(asg::ArithOp::Sub) => Ok(left - right),
                asg::BinaryOp::ArithOp(asg::ArithOp::Mul) => Ok(left * right),
                asg::BinaryOp::ArithOp(asg::ArithOp::Div) => Ok(left / right),
                asg::BinaryOp::ArithOp(asg::ArithOp::Mod)
                | asg::BinaryOp::ArithOp(asg::ArithOp::Rem) => Ok(left % right),
                asg::BinaryOp::ArithOp(asg::ArithOp::Shl) => Ok(left << right),
                asg::BinaryOp::ArithOp(asg::ArithOp::Shr) => Ok(left >> right),
                asg::BinaryOp::ArithOp(asg::ArithOp::BitXOr) => Ok(left ^ right),
                asg::BinaryOp::ArithOp(asg::ArithOp::BitAnd) => Ok(left & right),
                asg::BinaryOp::CmpOp(asg::CmpOp::Eq) => Ok((left == right) as i128),
                asg::BinaryOp::CmpOp(asg::CmpOp::Neq) => Ok((left != right) as i128),
                asg::BinaryOp::ConcatenationOp => Err(RustImportError::Unsupported(
                    "integer concatenation expressions are not supported".to_string(),
                )),
            }
        }
        asg::Expr::Cast(cast) => eval_int_expr(cast.operand(), symbols),
        other => Err(RustImportError::Unsupported(format!(
            "unsupported integer expression: {other:?}"
        ))),
    }
}

fn eval_float_expr(expr: &asg::TExpr, symbols: &SymbolTable) -> Result<f64, RustImportError> {
    match expr.expression() {
        asg::Expr::Literal(asg::Literal::Int(value)) => {
            let mag = *value.value() as f64;
            if *value.sign() {
                Ok(mag)
            } else {
                Ok(-mag)
            }
        }
        asg::Expr::Literal(asg::Literal::Float(value)) => value
            .value()
            .parse::<f64>()
            .map_err(|e| RustImportError::Message(format!("invalid float literal: {e}"))),
        asg::Expr::Identifier(id) => {
            let id = resolve_symbol(id)?;
            match symbols[&id].name() {
                "pi" | "π" => Ok(PI),
                "tau" | "τ" => Ok(TAU),
                "euler" | "ℇ" => Ok(E),
                other => Err(RustImportError::Unsupported(format!(
                    "identifier '{other}' is not a supported float constant"
                ))),
            }
        }
        asg::Expr::UnaryExpr(unary) => {
            let value = eval_float_expr(unary.operand(), symbols)?;
            match unary.op() {
                asg::UnaryOp::Minus => Ok(-value),
                asg::UnaryOp::Not | asg::UnaryOp::BitNot => Err(RustImportError::Unsupported(
                    "unsupported unary operator in float expression".to_string(),
                )),
            }
        }
        asg::Expr::BinaryExpr(binary) => {
            let left = eval_float_expr(binary.left(), symbols)?;
            let right = eval_float_expr(binary.right(), symbols)?;
            match binary.op() {
                asg::BinaryOp::ArithOp(asg::ArithOp::Add) => Ok(left + right),
                asg::BinaryOp::ArithOp(asg::ArithOp::Sub) => Ok(left - right),
                asg::BinaryOp::ArithOp(asg::ArithOp::Mul) => Ok(left * right),
                asg::BinaryOp::ArithOp(asg::ArithOp::Div) => Ok(left / right),
                asg::BinaryOp::ArithOp(asg::ArithOp::Mod)
                | asg::BinaryOp::ArithOp(asg::ArithOp::Rem) => Ok(left % right),
                asg::BinaryOp::ArithOp(asg::ArithOp::Shl)
                | asg::BinaryOp::ArithOp(asg::ArithOp::Shr)
                | asg::BinaryOp::ArithOp(asg::ArithOp::BitXOr)
                | asg::BinaryOp::ArithOp(asg::ArithOp::BitAnd)
                | asg::BinaryOp::CmpOp(_)
                | asg::BinaryOp::ConcatenationOp => Err(RustImportError::Unsupported(
                    "unsupported operator in float expression".to_string(),
                )),
            }
        }
        asg::Expr::Cast(cast) => eval_float_expr(cast.operand(), symbols),
        other => Err(RustImportError::Unsupported(format!(
            "unsupported float expression: {other:?}"
        ))),
    }
}

fn resolve_symbol(id: &SymbolIdResult) -> Result<SymbolId, RustImportError> {
    id.clone().map_err(|e| {
        RustImportError::Message(format!(
            "symbol resolution error while importing qasm3: {e:?}"
        ))
    })
}

fn broadcast_lists<T: Copy>(lists: &[Vec<T>]) -> Result<Vec<Vec<T>>, RustImportError> {
    if lists.is_empty() {
        return Ok(vec![Vec::new()]);
    }

    let extent = lists
        .iter()
        .map(Vec::len)
        .filter(|len| *len > 1)
        .max()
        .unwrap_or(1);

    for list in lists {
        let len = list.len();
        if !(len == 1 || len == extent) {
            return Err(RustImportError::Message(format!(
                "broadcast mismatch: expected each argument to have length 1 or {extent}, found {len}"
            )));
        }
    }

    let mut out = Vec::with_capacity(extent);
    for i in 0..extent {
        let mut tuple = Vec::with_capacity(lists.len());
        for list in lists {
            tuple.push(if list.len() == 1 { list[0] } else { list[i] });
        }
        out.push(tuple);
    }
    Ok(out)
}

fn broadcast_measure_pairs(
    qubits: &[Qubit],
    clbits: &[Clbit],
) -> Result<Vec<(Qubit, Clbit)>, RustImportError> {
    if qubits.is_empty() || clbits.is_empty() {
        return Err(RustImportError::Message(
            "measurement operands cannot be empty".to_string(),
        ));
    }

    let extent = qubits.len().max(clbits.len());
    if !(qubits.len() == 1 || qubits.len() == extent) {
        return Err(RustImportError::Message(
            "measurement qubit broadcast mismatch".to_string(),
        ));
    }
    if !(clbits.len() == 1 || clbits.len() == extent) {
        return Err(RustImportError::Message(
            "measurement clbit broadcast mismatch".to_string(),
        ));
    }

    let mut out = Vec::with_capacity(extent);
    for i in 0..extent {
        let q = if qubits.len() == 1 {
            qubits[0]
        } else {
            qubits[i]
        };
        let c = if clbits.len() == 1 {
            clbits[0]
        } else {
            clbits[i]
        };
        out.push((q, c));
    }
    Ok(out)
}

fn parse_standard_gate(name: &str) -> Option<StandardGate> {
    match name {
        "global_phase" => Some(StandardGate::GlobalPhase),
        "h" => Some(StandardGate::H),
        "id" => Some(StandardGate::I),
        "x" => Some(StandardGate::X),
        "y" => Some(StandardGate::Y),
        "z" => Some(StandardGate::Z),
        "p" => Some(StandardGate::Phase),
        "r" => Some(StandardGate::R),
        "rx" => Some(StandardGate::RX),
        "ry" => Some(StandardGate::RY),
        "rz" => Some(StandardGate::RZ),
        "s" => Some(StandardGate::S),
        "sdg" => Some(StandardGate::Sdg),
        "sx" => Some(StandardGate::SX),
        "sxdg" => Some(StandardGate::SXdg),
        "t" => Some(StandardGate::T),
        "tdg" => Some(StandardGate::Tdg),
        "u" => Some(StandardGate::U),
        "u1" => Some(StandardGate::U1),
        "u2" => Some(StandardGate::U2),
        "u3" => Some(StandardGate::U3),
        "ch" => Some(StandardGate::CH),
        "cx" | "CX" => Some(StandardGate::CX),
        "cy" => Some(StandardGate::CY),
        "cz" => Some(StandardGate::CZ),
        "dcx" => Some(StandardGate::DCX),
        "ecr" => Some(StandardGate::ECR),
        "swap" => Some(StandardGate::Swap),
        "iswap" => Some(StandardGate::ISwap),
        "cp" | "cphase" => Some(StandardGate::CPhase),
        "crx" => Some(StandardGate::CRX),
        "cry" => Some(StandardGate::CRY),
        "crz" => Some(StandardGate::CRZ),
        "cs" => Some(StandardGate::CS),
        "csdg" => Some(StandardGate::CSdg),
        "csx" => Some(StandardGate::CSX),
        "cu" => Some(StandardGate::CU),
        "cu1" => Some(StandardGate::CU1),
        "cu3" => Some(StandardGate::CU3),
        "rxx" => Some(StandardGate::RXX),
        "ryy" => Some(StandardGate::RYY),
        "rzz" => Some(StandardGate::RZZ),
        "rzx" => Some(StandardGate::RZX),
        "xx_minus_yy" => Some(StandardGate::XXMinusYY),
        "xx_plus_yy" => Some(StandardGate::XXPlusYY),
        "ccx" => Some(StandardGate::CCX),
        "ccz" => Some(StandardGate::CCZ),
        "cswap" => Some(StandardGate::CSwap),
        "rccx" => Some(StandardGate::RCCX),
        "mcx" => Some(StandardGate::C3X),
        "c3sx" => Some(StandardGate::C3SX),
        "rcccx" => Some(StandardGate::RC3X),
        _ => None,
    }
}
