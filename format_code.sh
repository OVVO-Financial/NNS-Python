#!/bin/bash
[ -z "$BLACK" ] && BLACK="black"
[ -z "$ISORT" ] && ISORT="isort"
[ -z "$AUTOFLAKE" ] && AUTOFLAKE="autoflake"

$AUTOFLAKE NNS/ tests/ --remove-all-unused-imports --remove-unused-variables --remove-duplicate-keys --expand-star-imports --recursive --in-place
$BLACK NNS/ tests/ --line-length=100 --target-version py37
$ISORT --multi-line=3 --trailing-comma --force-grid-wrap=0 --combine-as --line-width 100 --interactive NNS/ tests/
