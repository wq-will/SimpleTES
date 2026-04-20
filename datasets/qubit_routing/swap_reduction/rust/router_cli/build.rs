use std::env;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    println!("cargo:rerun-if-env-changed=VIRTUAL_ENV");
    println!("cargo:rerun-if-env-changed=UV_PROJECT_ENVIRONMENT");

    if let Some(libdir) = detect_python_libdir() {
        if !libdir.is_empty() {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{libdir}");
        }
    }
}

fn detect_python_libdir() -> Option<String> {
    let python = if let Ok(p) = env::var("PYO3_PYTHON") {
        p
    } else if let Ok(prefix) = env::var("CONDA_PREFIX") {
        format!("{prefix}/bin/python")
    } else if let Ok(prefix) = env::var("VIRTUAL_ENV") {
        format!("{prefix}/bin/python")
    } else if let Ok(prefix) = env::var("UV_PROJECT_ENVIRONMENT") {
        format!("{prefix}/bin/python")
    } else {
        "python3".to_string()
    };

    let output = Command::new(python)
        .args([
            "-c",
            "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or '')",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let libdir = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if libdir.is_empty() {
        None
    } else {
        Some(libdir)
    }
}
