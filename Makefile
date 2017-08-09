clean:
	rm -rf **/*.pyc

watch:
	guard

install:
	pip install . --upgrade
