run:
	python -m pcog pcog:__main__

clean:
	rm -rf **/*.pyc

log:
	tail -f pcog.log

test:
	python -m unnittest test.usm_test
