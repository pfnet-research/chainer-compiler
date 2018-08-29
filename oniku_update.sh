cd oniku/oniku
git fetch
git reset --hard origin/master
./setup.sh
cmake .
make
./scripts/runtests.py

