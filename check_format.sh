set -e
black ./src/*.py
flake8 ./src/*.py
pylint --extension-pkg-allow-list=torch,mpi4py ./src/*.py

exit

black ant_cants.py
flake8 ant_cants.py
pylint ant_cants.py

black rnn.py
flake8 rnn.py
pylint --extension-pkg-allow-list=torch  rnn.py

black colony_cants.py
flake8 colony_cants.py
pylint --extension-pkg-allow-list=mpi4py colony_cants.py
