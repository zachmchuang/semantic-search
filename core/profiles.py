"""Turn spreadsheet rows into labeled text for embedding."""

import pandas as pd


def _clean(value) -> str:
    """Normalize whitespace and convert missing values to an empty string."""
    if value is None or pd.isna(value):
        return ''
    return ' '.join(str(value).strip().split())


def _get_field(row: pd.Series, *names: str, default=''):
    """Try several possible column names and return the first non-empty value."""
    for name in names:
        if name in row.index:
            value = row[name]
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            return value
    return default


def field_value(row: pd.Series, name: str, default=''):
    """Read a single column from a row, treating NaN/None as default."""
    value = row.get(name, default)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    return value


def row_to_candidate_profile(row: pd.Series) -> str:
    """Format a candidate row as labeled profile text for embedding."""
    position = _clean(_get_field(row, 'Position', 'position'))
    moreinfo = _clean(_get_field(row, 'Moreinfo', 'moreinfo'))
    cv = _clean(_get_field(row, 'CV', 'cv'))
    looking_for = _clean(_get_field(row, 'Looking For', 'looking for'))
    highlights = _clean(_get_field(row, 'Highlights', 'highlights'))
    english_level = _clean(_get_field(row, 'English Level', 'english level'))
    primary_keyword = _clean(_get_field(row, 'Primary Keyword', 'primary keyword'))

    # CV and Moreinfo often overlap; pick the longer text as bio and avoid duplication.
    if not cv or cv == moreinfo:
        bio, extra_cv = moreinfo, ''
    elif not moreinfo:
        bio, extra_cv = cv, ''
    else:
        bio = cv if len(cv) > len(moreinfo) else moreinfo
        extra_cv = cv if cv != bio else ''

    exp = _get_field(row, 'Experience Years', 'experience years', default=None)
    if pd.notna(exp):
        years = float(exp)
        experience = f'{years:g} year' if years == 1 else f'{years:g} years'
    else:
        experience = ''

    sections = [
        f'Position: {position}' if position else '',
        f'Bio: {bio}' if bio else '',
        f'Looking for: {looking_for}' if looking_for else '',
        f'Highlights: {highlights}' if highlights else '',
        f'English level: {english_level}' if english_level else '',
        f'Experience: {experience}' if experience else '',
        f'Primary keyword: {primary_keyword}' if primary_keyword else '',
        f'CV: {extra_cv}' if extra_cv else '',
    ]
    # Join non-empty sections with newlines; skip blank lines entirely.
    return '\n'.join(section for section in sections if section)


def row_to_job_profile(row: pd.Series) -> str:
    """Format a job row as labeled profile text for embedding."""
    position = _clean(_get_field(row, 'Position', 'position'))
    description = _clean(
        _get_field(row, 'Long Description', 'long description', 'Description', 'description')
    )
    company = _clean(_get_field(row, 'Company Name', 'company name'))
    exp_years = _clean(_get_field(row, 'Exp Years', 'exp years'))
    primary_keyword = _clean(_get_field(row, 'Primary Keyword', 'primary keyword'))
    english_level = _clean(_get_field(row, 'English Level', 'english level'))
    published = _clean(_get_field(row, 'Published', 'published'))

    sections = [
        f'Position: {position}' if position else '',
        f'Description: {description}' if description else '',
        f'Company: {company}' if company else '',
        f'Experience required: {exp_years}' if exp_years else '',
        f'Primary keyword: {primary_keyword}' if primary_keyword else '',
        f'English level: {english_level}' if english_level else '',
        f'Published: {published}' if published else '',
    ]
    return '\n'.join(section for section in sections if section)
