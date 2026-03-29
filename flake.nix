{
  description = "Nix dev shells for various workflows";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        config.cudaSupport = true;
      };
    in
    {
      devShells.${system} = {
        # Default shell: Python/uv + CUDA native libs (for torch, numpy, etc.)
        # Usage: nix develop   OR   direnv allow (with `use flake` in .envrc)
        default = pkgs.mkShell {
          name = "python-cuda-dev";

          # Tip: find missing native libs with:
          # find .venv/ -name "*.so" | xargs ldd | grep "not found" | sort | uniq
          NIX_LD_LIBRARY_PATH =
            pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc # libstdc++
              pkgs.zlib # libz (for numpy)
              pkgs.libuv

              # CUDA (for torch)
              pkgs.cudaPackages.cuda_cudart
              pkgs.cudaPackages.cudnn
              pkgs.cudaPackages.cudatoolkit
              pkgs.cudaPackages.cuda_nvrtc
              pkgs.cudaPackages.cuda_cupti
            ]
            + ":/run/opengl-driver/lib:/run/opengl-driver-32/lib";

          NIX_LD = pkgs.lib.fileContents "${pkgs.stdenv.cc}/nix-support/dynamic-linker";

          packages = [ pkgs.uv ];
        };

        # Minimal shell: Python/uv only, no CUDA
        # Usage: nix develop .#minimal
        minimal = pkgs.mkShell {
          name = "python-minimal-dev";

          NIX_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc
            pkgs.zlib
          ];

          NIX_LD = pkgs.lib.fileContents "${pkgs.stdenv.cc}/nix-support/dynamic-linker";

          packages = [ pkgs.uv ];
        };
      };

      templates.default = {
        path = ./.;
        description = "Python/CUDA dev shell";
      };
    };
}
