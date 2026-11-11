.PHONY: test run scan licences

test:
	nice -n 19 uv run pytest -q

run:
	uv run safe-streets

# The pre-publish scan (Task 10): footage, identifiers, private addresses, secrets.
scan:
	python3 scripts/prepublish.py --history

licences:
	uv run python scripts/check_licences.py
