# Companies House Categories Reference

## Available Categories

The following categories are available for Companies House document searches:

| Display Name | API Value | Description |
|--------------|-----------|-------------|
| `Accounts` | `accounts` | Company accounts and financial statements |
| `Confirmation Stmt` | `confirmation-statement` | Annual confirmation statements |
| `Officers` | `officers` | Director and officer appointments/changes |
| `Capital` | `capital` | Share capital changes |
| `Charges` | `mortgage` | Mortgages and charges |
| `Insolvency` | `insolvency` | Insolvency proceedings |
| `PSC` | `persons-with-significant-control` | Persons with significant control |
| `Name Change` | `change-of-name` | Company name changes |
| `Reg. Office` | `registered-office-address` | Registered office address changes |

## Usage in Code

```python
# Correct way to use in multiselect
ch_selected_categories_multiselect = st.multiselect(
    "Document Categories:",
    options=list(CH_CATEGORIES.keys()),  # Use keys for display
    default=["Accounts", "Confirmation Stmt"],  # Use exact key names
    key="ch_categories_multiselect_main"
)

# Convert to API values
ch_selected_categories_api = [CH_CATEGORIES[cat] for cat in ch_selected_categories_multiselect]
```

## Important Notes

- **Display names** (keys) are used in the UI
- **API values** (values) are sent to Companies House API
- Always use the exact key names from `CH_CATEGORIES.keys()` for defaults
- The mapping is defined in `app.py` around line 277

## Common Mistakes to Avoid

❌ **Wrong default value:**
```python
default=["Accounts", "Annual Return/Confirmation Statement"]  # This doesn't exist
```

✅ **Correct default value:**
```python
default=["Accounts", "Confirmation Stmt"]  # This matches the key exactly
``` 