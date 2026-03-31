import json

path = r'e:/hw/course_work/experiments/notebooks/notebook793eada43a.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'MS-AR done. Shape' in src:
        print(f"Found in cell {i}")
        old = (
            '# Fit + predict\n'
            '# \u0412\u043d\u0438\u043c\u0430\u043d\u0438\u0435: \u043c\u0435\u0434\u043b\u0435\u043d\u043d\u043e (~3-5 \u043c\u0438\u043d \u043d\u0430 330 \u043e\u043a\u043e\u043d) \u2014 EM-\u0430\u043b\u0433\u043e\u0440\u0438\u0442\u043c \u0444\u0438\u0442\u0438\u0442\u0441\u044f \u043d\u0430 \u043a\u0430\u0436\u0434\u043e\u043c \u043e\u043a\u043d\u0435\n'
            'print("Fitting MS-AR on test windows (per-window EM)...")\n'
            'm_msar = MarkovSwitchingARModel(k_regimes=2, order=1, switching_variance=True)\n'
            'm_msar.fit(X_tr, y_tr)\n'
            'preds["MS-AR"] = m_msar.predict(X_test_bt)\n'
            'models["MS-AR"] = m_msar\n'
            "print(f\"MS-AR done. Shape: {preds['MS-AR'].shape}\")"
        )
        new = (
            '# Fit + predict \u043d\u0430 val \u0438 test\n'
            '# \u0412\u043d\u0438\u043c\u0430\u043d\u0438\u0435: \u043c\u0435\u0434\u043b\u0435\u043d\u043d\u043e (~3-5 \u043c\u0438\u043d \u043d\u0430 \u043a\u0430\u0436\u0434\u044b\u0439 \u0441\u043f\u043b\u0438\u0442) \u2014 EM-\u0430\u043b\u0433\u043e\u0440\u0438\u0442\u043c \u0444\u0438\u0442\u0438\u0442\u0441\u044f \u043d\u0430 \u043a\u0430\u0436\u0434\u043e\u043c \u043e\u043a\u043d\u0435\n'
            'print("Fitting MS-AR (per-window EM)...")\n'
            'm_msar = MarkovSwitchingARModel(k_regimes=2, order=1, switching_variance=True)\n'
            'm_msar.fit(X_tr, y_tr)\n'
            'print("  predicting val...")\n'
            'preds_val["MS-AR"] = m_msar.predict(X_val_bt)\n'
            'print("  predicting test...")\n'
            'preds["MS-AR"]     = m_msar.predict(X_test_bt)\n'
            'models["MS-AR"]    = m_msar\n'
            "print(f\"MS-AR done. val={preds_val['MS-AR'].shape}  test={preds['MS-AR'].shape}\")"
        )
        if old in src:
            new_src = src.replace(old, new)
            cell['source'] = [new_src]
            print("  Replaced OK")
        else:
            idx = src.find('MS-AR done')
            print("  No exact match. Context:")
            print(repr(src[max(0, idx-400):idx+60]))
        break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Saved.")
