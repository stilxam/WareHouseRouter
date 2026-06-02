{
  description = "UV Flake";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    unstable-nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, unstable-nixpkgs, flake-utils}:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system ;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        unstable-pkgs = import unstable-nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        cudatoolkit = pkgs.cudaPackages_12.cudatoolkit;
        uvFHSenv = pkgs.buildFHSEnv {
          name = "uv-environment";
          runScript = "bash";
          targetPkgs = pkgs: [
            pkgs.python313 
            
            pkgs.uv
            pkgs.cmake
            pkgs.ninja
            pkgs.tree-sitter
            cudatoolkit
            pkgs.nixd
            pkgs.nil
            pkgs.ruff
            pkgs.gcc
            pkgs.zlib
            pkgs.nodejs
           ];

           profile = ''
             export UV_PYTHON=python3.13
             
             if [ ! -d ".venv" ]; then
                echo "Creating Python 3.13 virtual environment..."
                uv venv .venv --python python3.13
             fi
             source .venv/bin/activate
             
             echo "Envir
	     '';
        };
      in
      {
        devShells.default = uvFHSenv.env; 
      });
}

