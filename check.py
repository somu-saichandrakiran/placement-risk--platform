import sys
print(f'Python: {sys.version}\n')

failed = []
pkgs = ['pandas','numpy','sklearn','xgboost','shap',
        'mlflow','fastapi','sqlalchemy','loguru',
        'pydantic','plotly','streamlit']

for pkg in pkgs:
    try:
        __import__(pkg)
        print(f'  OK   {pkg}')
    except Exception as e:
        print(f'  FAIL {pkg} — {e}')
        failed.append(pkg)

print()
if not failed:
    print('READY TO BUILD')
else:
    print(f'NEED TO FIX: {failed}')