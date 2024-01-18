install_poetry
	curl -sSL https://install.python-poetry.org | python3 -

install_environment
	poetry install
	poetry shell
	sudo apt-get update && sudo apt-get install -y libsndfile1 ffmpeg
	pip install nemo_toolkit[all]==1.21