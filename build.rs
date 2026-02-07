use std::env;
use std::path::PathBuf;

fn main() {
    if env::var("DOCS_RS").is_ok() {
        return;
    }

    let (_include_path, lib_path) = if let Ok(dir) = env::var("OIDN_DIR") {
        let root = PathBuf::from(&dir);
        let include = root.join("include");
        // Try common locations for the OIDN import library (Windows: .lib, Unix: .a or .so)
        let lib = ["lib", "Release", "Debug"]
            .iter()
            .map(|p| root.join(p))
            .find(|p| p.exists())
            .unwrap_or_else(|| root.clone());
        (include, lib)
    } else {
        match pkg_config::Config::new().probe("OpenImageDenoise") {
            Ok(lib) => {
                let include = lib.include_paths.first().cloned().unwrap_or_else(PathBuf::new);
                let lib_path = lib.link_paths.first().cloned().unwrap_or_else(PathBuf::new);
                (include, lib_path)
            }
            Err(e) => {
                println!("cargo:warning=oidn-wgpu: OpenImageDenoise not found via pkg-config: {}", e);
                println!(
                    "cargo:warning=Set OIDN_DIR to the install directory (containing include/ and lib/) or install OIDN and pkg-config."
                );
                panic!("OpenImageDenoise required. Set OIDN_DIR or use pkg-config.");
            }
        }
    };

    println!("cargo:rustc-link-search=native={}", lib_path.display());
    println!("cargo:rustc-link-lib=OpenImageDenoise");
    println!("cargo:rerun-if-env-changed=OIDN_DIR");
}
