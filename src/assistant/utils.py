def format_legacy_record_markdown(field, value):
    """Format legacy metadata record into a nicely formatted Markdown string.

    Args:
        field: field name
        value: value associated with the field
    """

    return f"""
**Legacy field**: {field}
**Legacy value**: {value}
"""
