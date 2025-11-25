{
  description = "PRUN: Playground for Pruning Algorithms";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.hl = {
    url = "github:pamburus/hl";
    inputs.nixpkgs.follows = "nixpkgs";
  };
  inputs.flake-utils = {
    url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    hl,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {inherit system;};
        hl-bin = hl.packages.${system}.default;
        # get a python build with optimizations enabled, following
        # this suggestion: https://discourse.nixos.org/t/why-is-the-nix-compiled-python-slower/18717/9
        python = pkgs.python312;
        # .override {
        #   enableOptimizations = true;
        #   reproducibleBuild = false;
        #   self = python;
        # };

        container = pkgs.singularity-tools.buildImage {
          name = "prun-container";
          runScript = "#!${pkgs.stdenv.shell}\npython $@";
          contents = [
            (python.withPackages
              (ppkgs:
                with ppkgs; [
                  numpy
                  pandas
                  icecream
                  scikit-learn
                  scipy
                  matplotlib
                  seaborn
                  llmcompressor
                  transformers
                  vllm
                ]))
          ];
          diskSize = 1024 * 3; # necessary to fit the packages, otherwise the build fails
        };
      in {
        packages.container = container;
        devShells.default = (pkgs.mkShell.override {stdenv = pkgs.clangStdenv;}) {
          venvDir = ".venv";

          packages = with pkgs; [
            gcc
            lldb
            clang-tools
            (
              python.withPackages
              (ps:
                with ps; [
                  venvShellHook
                  numpy
                  pandas
                  matplotlib
                  seaborn
                  tornado
                  umap-learn
                  h5py
                  nanobind
                  icecream
                  llmcompressor
                  transformers
                  vllm
                  scikit-build-core
                ])
            )
            hdf5
            sqlite-interactive
            cmake
            just
            bear # To generate compile_commands.json files
            llvmPackages.openmp
            llvmPackages.libcxx
            rr
            gdbgui
            valgrind
            highfive
            samply
            boost
            cereal
            catch2_3
            ensmallen
            armadillo
            hl-bin
          ];

          NIX_ENFORCE_NO_NATIVE = false;
        };
      }
    );
}