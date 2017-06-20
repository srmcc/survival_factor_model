{pkgs ? import <nixpkgs> {}}:

with pkgs;

let

  survEnv = stdenv.mkDerivation rec {
    name = "survEnv";

    phases = ["fixupPhase"];

    #TODO: override pkgs to get seaborn
    pythonPkgs = python3.withPackages (ps: with ps; [numpy scipy scikitlearn bokeh matplotlib pandas sphinx ggplot ]);

    buildInputs = [pythonPkgs stdenv];

    src = ./.;

    passthru = {
     dockerImg = dockerTools.buildImage {
       name = "surv";
       contents = pythonPkgs;
     };
    };


  };
in
  survEnv.passthru.dockerImg
