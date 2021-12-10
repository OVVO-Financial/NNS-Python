#!/bin/bash
[ -z "$PYTEST" ] && PYTEST="python -m pytest"
[ -z "$PIP" ] && PIP="python -m pip"

echo "Installing NNS as development package, $PIP"
$PIP install --no-cache-dir --user -e .[test]
echo

echo "Starting pytest: $PYTEST"
$PYTEST --cache-clear --cov=NNS --cov-report term-missing --cov-fail-under=70 ./tests/
