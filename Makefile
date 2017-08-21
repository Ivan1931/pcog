clean:
	rm -rf **/*.pyc

watch:
	guard

install:
	pip install . --upgrade

run:
	python -m pcog pcog:__main__