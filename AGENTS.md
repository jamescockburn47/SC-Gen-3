# Repository Guidelines

This project is a Streamlit application with supporting utilities and tests.
Please follow these conventions when contributing:

## Development
- Use **Python 3.10+**.
- Install dependencies with `pip install -r requirements.txt` while network
  access is available. The file pins exact versions, including `pytest`, so
  this step must happen before any network restrictions.
- Unit tests are located in files starting with `test_`. Run them using:
  ```bash
  python -m unittest discover -v
  ```
  Ensure all tests pass before committing.
- Add new tests when fixing bugs or adding features.

## Style
- Follow standard **PEP8** style: four spaces per indent and reasonably
  short lines.
- Use type hints where practical and keep logging consistent with the
  existing modules (via `logging.getLogger(__name__)`).

## Commit Messages
- Use short imperative messages such as `Add new OCR option` or
  `Fix Google Drive loading`.
- Describe why the change is needed when the reason is not obvious.

## Running the App
- Create a `.env` file with the required environment variables as
  outlined in `README.md`.
- Launch the application with:
  ```bash
  streamlit run app.py
  ```


