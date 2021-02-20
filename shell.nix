with import <nixpkgs> {};
mkShell {
    buildInputs = [
        (python38.withPackages(ps: with ps; [
            flake8
            matplotlib
            numpy
            scikitlearn
            seaborn
        ]))
        feh
        shellcheck
    ];
    shellHook = ''
        . .shellhook
    '';
}
