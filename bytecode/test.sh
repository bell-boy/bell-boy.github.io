cd ./src
g++ ../tests.cpp parser.cpp parser.h transpiler.cpp transpiler.h -o ./tests.out && ./tests.out
rm ./tests.out