#import "../state.typ": bib_state
#context bib_state.get()


= Результаты <sec-results>

== Оценка качества предсказательной способности моделей

== Модели без кластеризации

Для моделей, обученных на полных данных без кластеризации, были получены следующие оптимальные конфигурации гиперпараметров:

#figure(
  table(
    columns: (2fr, 1.4fr, 1fr),
    align: (left, left, center),
    inset: (x: 9pt, y: 6pt),
    stroke: none,
    table.hline(stroke: 1pt + black),
    [*Модель*], [*Гиперпараметр*], [*Значение*],
    table.hline(stroke: 0.6pt + luma(120)),

    table.cell(rowspan: 5)[*LSTM*],
    [hidden\_size],  [160],
    [num\_layers],   [3],
    [epochs],        [50],
    [batch\_size],   [128],
    [lr],            [1.57 × 10#super[−4]],

    table.hline(stroke: 0.4pt + luma(180)),

    table.cell(rowspan: 5)[*GRU*],
    [hidden\_size],  [192],
    [num\_layers],   [2],
    [epochs],        [80],
    [batch\_size],   [128],
    [lr],            [2.37 × 10#super[−4]],

    table.hline(stroke: 0.4pt + luma(180)),

    table.cell(rowspan: 4)[*CNN*],
    [num\_filters],  [64],
    [epochs],        [70],
    [batch\_size],   [128],
    [lr],            [2.99 × 10#super[−3]],

    table.hline(stroke: 0.4pt + luma(180)),

    table.cell(rowspan: 4)[*Transformer*],
    [d\_model],      [224],
    [epochs],        [50],
    [batch\_size],   [128],
    [lr],            [1.78 × 10#super[−3]],

    table.hline(stroke: 0.4pt + luma(180)),

    table.cell(rowspan: 7)[*LightGBM*],
    [n\_estimators],          [600],
    [learning\_rate],         [0.158],
    [num\_leaves],            [95],
    [max\_depth],             [8],
    [min\_child\_samples],    [45],
    [subsample],              [0.552],
    [colsample\_bytree],      [0.519],

    table.hline(stroke: 1pt + black),
  ),
  caption: [Оптимальные гиперпараметры моделей],
)

// После подбора гиперпараметров все модели были оценены на тестовой выборке (71 247 окон). Результаты представлены в таблице ниже.

#figure(
  table(
    columns: (1.6fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center, center, center),
    inset: (x: 8pt, y: 6pt),
    stroke: none,
    table.hline(stroke: 1pt + black),
    [*Модель*], [*MAE*], [*RMSE*], [*MSE*], [*R²*], [*IC ↑*], [*Dir. Acc.*],
    table.hline(stroke: 0.6pt + luma(120)),
    [LSTM],        [0.01350], [0.02159], [0.000466], [−0.0097], [*0.0435*], [0.4918],
    [GRU],         [0.01352], [0.02154], [0.000464], [−0.0057], [*0.0644*], [0.4885],
    [CNN],         [0.01333], [0.02149], [0.000462], [−0.0005], [−0.0043],  [0.4652],
    [Transformer], [0.01333], [0.02149], [0.000462], [−0.0001], [0.0109],   [0.4751],
    [LightGBM],    [0.01333], [0.02149], [0.000462], [−0.0005], [0.0172],   [0.4775],
    table.hline(stroke: 0.4pt + luma(180)),
    [MS-AR],       [0.01341], [0.02163], [0.000468], [−0.0141], [0.0318],   [0.4844],
    [ARIMA],       [0.01368], [0.02201], [0.000485], [−0.0512], [−0.0071],  [0.4619],
    table.hline(stroke: 1pt + black),
  ),
  caption: [Метрики качества моделей на тестовой выборке],
)
// 
Модели демонстрируют идентичные значения MAE, MSE и RMSE.

Метрика *R^2* у всех моделей отрицательна, что характерно для финансовых временных рядов, так как модели не могут обьяснить диспрсию доходностей.

Так что отдельное внимание уделяется *IC* (Information Coefficient) — ранговая корреляция предсказаных и реальных доходностей. И *Dir. Accuracy* - доля правильно угаданных направлений движения.

*LSTM* демонстрирует второй по качеству результат по метрике *IC* = 0.044, однако превосходит оставльные модели по *Dir. Accuracy* = 0.4918.

Это может означать, что модель плохо предсказывоет мелкие изменения цены, но хорошо улавливает крупные, что намного важнее для постороения торговой стратегии.
// 
*GRU* показывает лучший результат по метрике *IC* = 0.064, что указывает на наибольшую согласованность направления предсказанных и реальных движений, однако по *Dir. Accuracy* = 0.4885 уступает *LSTM* на 0.0033. 

*MS-AR* занимает четвёртое место по IC = 0.032 — несмотря на классическую природу, явное моделирование режимов рынка даёт ненулевую предсказательную способность.

*ARIMA* и *CNN* имеют отрицательный IC, что свидетельствует об отсутствии значимой корреляции прогнозов с реальными доходностями на данном горизонте.


== Модели с кластеризацией

При кластерном подходе для каждого кластера с помощью Optuna подбираются собственные гиперпараметры и обучается отдельный набор из пяти моделей.
Тестовые окна классифицируются в кластер, после чего оцениваются соответствующей специализированной моделью.
Для каждого кластера приведены две таблицы: гиперпараметры кластерных моделей и сравнение метрик, где *кластерные* и *baseline*-модели (обученные на всём датасете) представлены отдельными блоками.


=== TS2Vec + KMeans 

TS2Vec + KMeans выделяет 5 кластеров:
 
 *C0* — боковик
 
  *C1* — бычий тренд
  
  *C2* — высокая волатильность
  
  *C3* — медвежий тренд
  
  *C4* — аномальные движения

==== C0 — нейтральный (боковое движение)

#figure(
  table(
    columns: (1.6fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center, center),
    inset: (x: 7pt, y: 5pt),
    stroke: none,
    table.hline(stroke: 1pt + black),
    [*Модель*], [*hidden_dim*], [*layers*], [*lr*], [*epochs*], [*batch*],
    table.hline(stroke: 0.6pt + luma(120)),
    [LSTM],        [96],  [2], [2.3×10#super[−4]], [20], [128],
    [GRU],         [112], [2], [1.8×10#super[−4]], [30], [128],
    [CNN],         [48],  [—], [3.2×10#super[−3]], [15], [128],
    [Transformer], [160], [—], [1.9×10#super[−3]], [50], [128],
    [LightGBM],    [—],   [—], [—], [—], [—],
    table.hline(stroke: 1pt + black),
  ),
  caption: [Гиперпараметры кластерных моделей — TS2Vec C0],
)

#figure(
  table(
    columns: (1fr, 1.8fr, 1fr, 1fr, 1fr),
    align: (left, left, center, center, center),
    inset: (x: 7pt, y: 5pt),
    stroke: none,
    table.hline(stroke: 1pt + black),
    [*Тип*], [*Модель*], [*MAE*], [*IC ↑*], [*Dir.Acc.*],
    table.hline(stroke: 0.6pt + luma(120)),
    table.cell(rowspan: 5)[Кластерная],
    [LSTM],        [0.01352], [0.0334], [0.4841],
    [GRU],         [0.01341], [0.0448], [0.4901],
    [CNN],         [0.01378], [0.0201], [0.4762],
    [Transformer], [0.01359], [0.0298], [0.4812],
    [LightGBM],    [0.01344], [0.0319], [0.4798],
    table.hline(stroke: 0.6pt + luma(120)),
    table.cell(rowspan: 5)[Baseline],
    [*LSTM*],        [0.01331], [*0.0521*], [*0.4961*],
    [GRU],           [0.01344],   [0.0462],   [0.4889],
    [CNN],           [0.01339],   [0.0148],   [0.4751],
    [Transformer],   [0.01336],   [0.0271],   [0.4831],
    [*LightGBM*],    [0.01328], [0.0489], [0.4941],
    table.hline(stroke: 1pt + black),
  ),
  caption: [Метрики — TS2Vec C0: кластерные vs baseline],
)

На нейтральном кластере кластерные модели не превосходят baseline модели по метрикам, так как боковое движение не содержит никаких устойчивых патернов и baseline модели просто видели больше таких примеров.

==== C1 — бычий тренд

#figure(
  table(
    columns: (1.6fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center, center),
    inset: (x: 7pt, y: 5pt),
    stroke: none,
    table.hline(stroke: 1pt + black),
    [*Модель*], [*hidden_dim*], [*layers*], [*lr*], [*epochs*], [*batch*],
    table.hline(stroke: 0.6pt + luma(120)),
    [LSTM],        [128], [2], [2.1×10#super[−4]], [25], [128],
    [GRU],         [160], [2], [1.7×10#super[−4]], [30], [128],
    [CNN],         [64],  [—], [2.5×10#super[−3]], [20], [128],
    [Transformer], [192], [—], [1.5×10#super[−3]], [55], [128],
    [LightGBM],    [—],   [—], [—], [—], [—],
    table.hline(stroke: 1pt + black),
  ),
  caption: [Гиперпараметры кластерных моделей — TS2Vec C1],
)

#figure(
  table(
    columns: (1fr, 1.8fr, 1fr, 1fr, 1fr),
    align: (left, left, center, center, center),
    inset: (x: 7pt, y: 5pt),
    stroke: none,
    table.hline(stroke: 1pt + black),
    [*Тип*], [*Модель*], [*MAE*], [*IC ↑*], [*Dir.Acc.*],
    table.hline(stroke: 0.6pt + luma(120)),
    table.cell(rowspan: 5)[Кластерная],
    [LSTM],        [0.01291], [0.0712],   [0.5143],
    [GRU],         [0.01279], [*0.0891*], [*0.5221*],
    [CNN],         [0.01318], [0.0531],   [0.5019],
    [Transformer], [0.01301], [0.0678],   [0.5097],
    [LightGBM],    [0.01278], [0.0803],   [0.5072],
    table.hline(stroke: 0.6pt + luma(120)),
    table.cell(rowspan: 5)[Baseline],
    [LSTM],        [0.01318], [0.0601], [0.5051],
    [GRU],         [0.01324], [0.0734], [0.5018],
    [CNN],         [0.01341], [0.0489], [0.4951],
    [Transformer], [0.01329], [0.0541], [0.4982],
    [LightGBM],    [0.01311], [0.0691], [0.5001],
    table.hline(stroke: 1pt + black),
  ),
  caption: [Метрики — TS2Vec C1: кластерные vs baseline],
)

На кластере с бычьим трендом кластерные модели превосходят baseline модели по метрикам, так как модели кластерные модели лучше уловили эту особенность из-за большего процента таких примеров.

==== C2 — высокая волатильность

#figure(
  table(
    columns: (1.6fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center, center),
    inset: (x: 7pt, y: 5pt),
    stroke: none,
    table.hline(stroke: 1pt + black),
    [*Модель*], [*hidden_dim*], [*layers*], [*lr*], [*epochs*], [*batch*],
    table.hline(stroke: 0.6pt + luma(120)),
    [LSTM],        [128], [3], [1.9×10#super[−4]], [22], [128],
    [GRU],         [160], [2], [2.1×10#super[−4]], [19], [128],
    [CNN],         [64],  [—], [2.8×10#super[−3]], [26], [128],
    [Transformer], [224], [—], [1.7×10#super[−3]], [21], [128],
    [LightGBM],    [—],   [—], [—], [480], [—],
    table.hline(stroke: 1pt + black),
  ),
  caption: [Гиперпараметры кластерных моделей — TS2Vec C2],
)

#figure(
  table(
    columns: (1fr, 1.8fr, 1fr, 1fr, 1fr),
    align: (left, left, center, center, center),
    inset: (x: 7pt, y: 5pt),
    stroke: none,
    table.hline(stroke: 1pt + black),
    [*Тип*], [*Модель*], [*MAE*], [*IC ↑*], [*Dir.Acc.*],
    table.hline(stroke: 0.6pt + luma(120)),
    table.cell(rowspan: 5)[Кластерная],
    [LSTM],           [0.01418], [0.0089], [0.4718],
    [GRU],            [0.01401], [0.0112], [0.4739],
    [CNN],            [0.01437], [−0.0061], [0.4664],
    [Transformer],    [0.01409], [0.0074], [0.4701],
    [*LightGBM*],     [0.01362], [*0.0341*], [*0.4891*],
    table.hline(stroke: 0.6pt + luma(120)),
    table.cell(rowspan: 5)[Baseline],
    [LSTM],        [0.01371], [0.0241], [0.4831],
    [GRU],         [0.01364], [0.0289], [0.4848],
    [CNN],         [0.01358], [0.0071], [0.4712],
    [Transformer], [0.01361], [0.0159], [0.4769],
    [LightGBM],    [0.01354], [0.0218], [0.4801],
    table.hline(stroke: 1pt + black),
  ),
  caption: [Метрики — TS2Vec C2: кластерные vs baseline],
)

На кластере с высокой волатильностью кластерный *LightGBM* показывает лучший IC = 0.034, тогда как нейросетевые модели, специализированные на этом кластере, уступают даже своим baseline-аналогам.

==== C3 — медвежий тренд

#figure(
  table(
    columns: (1.6fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center, center),
    inset: (x: 7pt, y: 5pt),
    stroke: none,
    table.hline(stroke: 1pt + black),
    [*Модель*], [*hidden_dim*], [*layers*], [*lr*], [*epochs*], [*batch*],
    table.hline(stroke: 0.6pt + luma(120)),
    [LSTM],        [128], [2], [1.6×10#super[−4]], [24], [128],
    [GRU],         [144], [2], [2.0×10#super[−4]], [21], [128],
    [CNN],         [64],  [—], [2.4×10#super[−3]], [27], [128],
    [Transformer], [192], [—], [1.6×10#super[−3]], [22], [128],
    [LightGBM],    [—],   [—], [—], [440], [—],
    table.hline(stroke: 1pt + black),
  ),
  caption: [Гиперпараметры кластерных моделей — TS2Vec C3],
)


#figure(
  table(
    columns: (1fr, 1.8fr, 1fr, 1fr, 1fr),
    align: (left, left, center, center, center),
    inset: (x: 7pt, y: 5pt),
    stroke: none,
    table.hline(stroke: 1pt + black),
    [*Тип*], [*Модель*], [*MAE*], [*IC ↑*], [*Dir.Acc.*],
    table.hline(stroke: 0.6pt + luma(120)),
    table.cell(rowspan: 5)[Кластерная],
    [LSTM],        [0.01298], [*0.0841*], [0.5121],
    [GRU],         [0.01311], [0.0764],   [*0.5198*],
    [CNN],         [0.01334], [0.0581],   [0.5009],
    [Transformer], [0.01319], [0.0712],   [0.5087],
    [LightGBM],    [0.01307], [0.0698],   [0.5041],
    table.hline(stroke: 0.6pt + luma(120)),
    table.cell(rowspan: 5)[Baseline],
    [LSTM],        [0.01389], [0.0312], [0.4851],
    [GRU],         [0.01394], [0.0491], [0.4821],
    [CNN],         [0.01378], [0.0041], [0.4701],
    [Transformer], [0.01381], [0.0188], [0.4791],
    [LightGBM],    [0.01371], [0.0398], [0.4812],
    table.hline(stroke: 1pt + black),
  ),
  caption: [Метрики — TS2Vec C3: кластерные vs baseline],
)
Анологично кластеру 1 кластерные модели превосходят baseline модели по метрикам, так как лучше уловили особенность окна из-за большего процента таких примеров.

==== C4 — аномальные движения

#figure(
  table(
    columns: (1.6fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center, center),
    inset: (x: 7pt, y: 5pt),
    stroke: none,
    table.hline(stroke: 1pt + black),
    [*Модель*], [*hidden_dim*], [*layers*], [*lr*], [*epochs*], [*batch*],
    table.hline(stroke: 0.6pt + luma(120)),
    [LSTM],        [96],  [2], [2.4×10#super[−4]], [18], [128],
    [GRU],         [128], [2], [1.9×10#super[−4]], [16], [128],
    [CNN],         [48],  [—], [2.9×10#super[−3]], [23], [128],
    [Transformer], [160], [—], [1.7×10#super[−3]], [20], [128],
    [LightGBM],    [—],   [—], [—], [360], [—],
    table.hline(stroke: 1pt + black),
  ),
  caption: [Гиперпараметры кластерных моделей — TS2Vec C4],
)



#figure(
  table(
    columns: (1fr, 1.8fr, 1fr, 1fr, 1fr),
    align: (left, left, center, center, center),
    inset: (x: 7pt, y: 5pt),
    stroke: none,
    table.hline(stroke: 1pt + black),
    [*Тип*], [*Модель*], [*MAE*], [*IC ↑*], [*Dir.Acc.*],
    table.hline(stroke: 0.6pt + luma(120)),
    table.cell(rowspan: 5)[Кластерная],
    [LSTM],        [0.01319], [0.0487],   [0.5041],
    [GRU],         [0.01308], [*0.0631*], [0.5089],
    [CNN],         [0.01331], [0.0312],   [0.4941],
    [Transformer], [0.01314], [0.0558],   [*0.5147*],
    [LightGBM],    [0.01311], [0.0521],   [0.5018],
    table.hline(stroke: 0.6pt + luma(120)),
    table.cell(rowspan: 5)[Baseline],
    [LSTM],        [0.01361], [0.0198], [0.4801],
    [GRU],         [0.01348], [0.0341], [0.4834],
    [CNN],         [0.01372], [−0.0081], [0.4641],
    [Transformer], [0.01358], [0.0089], [0.4718],
    [LightGBM],    [0.01349], [0.0241], [0.4769],
    table.hline(stroke: 1pt + black),
  ),
  caption: [Метрики — TS2Vec C4: кластерные vs baseline],
)

В условиях аномальных движений кластерные модели превосходят baseline модели по метрикам.

=== Handmade + KMeans

Handmade + KMeans выделяет 4 кластера: 

*C0* — боковик

*C1* — бычий тренд

*C2* — нисходящий с отрицательной асимметрией

*C3* — кризисный (высокая волатильность + падение)

==== C0 — нейтральный (боковое движение)

#figure(
  table(
    columns: (1.6fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center, center),
    inset: (x: 7pt, y: 5pt),
    stroke: none,
    table.hline(stroke: 1pt + black),
    [*Модель*], [*hidden_dim*], [*layers*], [*lr*], [*epochs*], [*batch*],
    table.hline(stroke: 0.6pt + luma(120)),
    [LSTM],        [112], [2], [1.9×10#super[−4]], [31], [128],
    [GRU],         [128], [2], [2.1×10#super[−4]], [28], [128],
    [CNN],         [48],  [—], [3.0×10#super[−3]], [35], [128],
    [Transformer], [160], [—], [1.8×10#super[−3]], [33], [128],
    [LightGBM],    [—],   [—], [—], [520], [—],
    table.hline(stroke: 1pt + black),
  ),
  caption: [Гиперпараметры кластерных моделей — Handmade C0],
)

#figure(
  table(
    columns: (1fr, 1.8fr, 1fr, 1fr, 1fr),
    align: (left, left, center, center, center),
    inset: (x: 7pt, y: 5pt),
    stroke: none,
    table.hline(stroke: 1pt + black),
    [*Тип*], [*Модель*], [*MAE*], [*IC ↑*], [*Dir.Acc.*],
    table.hline(stroke: 0.6pt + luma(120)),
    table.cell(rowspan: 5)[Кластерная],
    [LSTM],        [0.01358], [0.0329], [0.4831],
    [GRU],         [0.01344], [0.0381], [0.4871],
    [CNN],         [0.01371], [0.0198], [0.4748],
    [Transformer], [0.01352], [0.0274], [0.4801],
    [LightGBM],    [0.01341], [0.0312], [0.4818],
    table.hline(stroke: 0.6pt + luma(120)),
    table.cell(rowspan: 5)[Baseline],
    [*LSTM*],        [0.01339], [*0.0548*], [0.4911],
    [GRU],           [0.01341], [0.0491],    [*0.4948*],
    [CNN],           [0.01348], [0.0141],   [0.4762],
    [Transformer],   [0.01344], [0.0318],   [0.4841],
    [LightGBM],      [0.01336], [0.0421],   [0.4891],
    table.hline(stroke: 1pt + black),
  ),
  caption: [Метрики — Handmade C0: кластерные vs baseline],
)

Baseline модели видели больше примеров нейтральных окон и выигрывают у кластерных моделей.

==== C1 — бычий тренд

#figure(
  table(
    columns: (1.6fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center, center),
    inset: (x: 7pt, y: 5pt),
    stroke: none,
    table.hline(stroke: 1pt + black),
    [*Модель*], [*hidden_dim*], [*layers*], [*lr*], [*epochs*], [*batch*],
    table.hline(stroke: 0.6pt + luma(120)),
    [LSTM],        [128], [2], [1.8×10#super[−4]], [29], [128],
    [GRU],         [144], [2], [2.2×10#super[−4]], [32], [128],
    [CNN],         [64],  [—], [2.7×10#super[−3]], [36], [128],
    [Transformer], [160], [—], [1.6×10#super[−3]], [31], [128],
    [LightGBM],    [—],   [—], [—], [500], [—],
    table.hline(stroke: 1pt + black),
  ),
  caption: [Гиперпараметры кластерных моделей — Handmade C1],
)

#figure(
  table(
    columns: (1fr, 1.8fr, 1fr, 1fr, 1fr),
    align: (left, left, center, center, center),
    inset: (x: 7pt, y: 5pt),
    stroke: none,
    table.hline(stroke: 1pt + black),
    [*Тип*], [*Модель*], [*MAE*], [*IC ↑*], [*Dir.Acc.*],
    table.hline(stroke: 0.6pt + luma(120)),
    table.cell(rowspan: 5)[Кластерная],
    [LSTM],        [0.01304], [0.0718],   [0.5098],
    [GRU],         [0.01289], [*0.0901*], [0.5171],
    [CNN],         [0.01321], [0.0489],   [0.4991],
    [Transformer], [0.01301], [0.0631],   [*0.5214*],
    [LightGBM],    [0.01294], [0.0784],   [0.5041],
    table.hline(stroke: 0.6pt + luma(120)),
    table.cell(rowspan: 5)[Baseline],
    [LSTM],        [0.01321], [0.0524], [0.5001],
    [GRU],         [0.01318], [0.0648], [0.4971],
    [CNN],         [0.01334], [0.0438], [0.4931],
    [Transformer], [0.01324], [0.0511], [0.4962],
    [LightGBM],    [0.01309], [0.0601], [0.4941],
    table.hline(stroke: 1pt + black),
  ),
  caption: [Метрики — Handmade C1: кластерные vs baseline],
)

Больший процент примеров бычьего тренда позволяет кластерным моделям превосходить baseline модели по метрикам.

==== C2 — нисходящий с отрицательной асимметрией

#figure(
  table(
    columns: (1.6fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center, center),
    inset: (x: 7pt, y: 5pt),
    stroke: none,
    table.hline(stroke: 1pt + black),
    [*Модель*], [*hidden_dim*], [*layers*], [*lr*], [*epochs*], [*batch*],
    table.hline(stroke: 0.6pt + luma(120)),
    [LSTM],        [112], [2], [2.0×10#super[−4]], [27], [128],
    [GRU],         [128], [2], [1.9×10#super[−4]], [24], [128],
    [CNN],         [64],  [—], [2.6×10#super[−3]], [31], [128],
    [Transformer], [160], [—], [1.7×10#super[−3]], [28], [128],
    [LightGBM],    [—],   [—], [—], [490], [—],
    table.hline(stroke: 1pt + black),
  ),
  caption: [Гиперпараметры кластерных моделей — Handmade C2],
)



#figure(
  table(
    columns: (1fr, 1.8fr, 1fr, 1fr, 1fr),
    align: (left, left, center, center, center),
    inset: (x: 7pt, y: 5pt),
    stroke: none,
    table.hline(stroke: 1pt + black),
    [*Тип*], [*Модель*], [*MAE*], [*IC ↑*], [*Dir.Acc.*],
    table.hline(stroke: 0.6pt + luma(120)),
    table.cell(rowspan: 5)[Кластерная],
    [LSTM],        [0.01348], [0.0211], [0.4778],
    [GRU],         [0.01341], [0.0248], [0.4812],
    [CNN],         [0.01364], [0.0041], [0.4701],
    [Transformer], [0.01351], [0.0178], [0.4758],
    [LightGBM],    [0.01329], [*0.0491*], [*0.4921*],
    table.hline(stroke: 0.6pt + luma(120)),
    table.cell(rowspan: 5)[Baseline],
    [LSTM],        [0.01378], [0.0281], [0.4871],
    [GRU],         [0.01381], [0.0441], [0.4841],
    [CNN],         [0.01362], [−0.0018], [0.4671],
    [Transformer], [0.01369], [0.0148], [0.4748],
    [LightGBM],    [0.01358], [0.0271], [0.4781],
    table.hline(stroke: 1pt + black),
  ),
  caption: [Метрики — Handmade C2: кластерные vs baseline],
)

Кластерный *LightGBM* показывает лучший результат по IC = 0.049, так как хорошо ловит ассиметричные распределения через пороговые разбиения.

==== C3 — кризисный режим

#figure(
  table(
    columns: (1.6fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center, center),
    inset: (x: 7pt, y: 5pt),
    stroke: none,
    table.hline(stroke: 1pt + black),
    [*Модель*], [*hidden_dim*], [*layers*], [*lr*], [*epochs*], [*batch*],
    table.hline(stroke: 0.6pt + luma(120)),
    [LSTM],        [128], [2], [1.7×10#super[−4]], [30], [128],
    [GRU],         [144], [2], [2.0×10#super[−4]], [34], [128],
    [CNN],         [56],  [—], [2.8×10#super[−3]], [33], [128],
    [Transformer], [176], [—], [1.5×10#super[−3]], [29], [128],
    [LightGBM],    [—],   [—], [—], [510], [—],
    table.hline(stroke: 1pt + black),
  ),
  caption: [Гиперпараметры кластерных моделей — Handmade C3],
)



#figure(
  table(
    columns: (1fr, 1.8fr, 1fr, 1fr, 1fr),
    align: (left, left, center, center, center),
    inset: (x: 7pt, y: 5pt),
    stroke: none,
    table.hline(stroke: 1pt + black),
    [*Тип*], [*Модель*], [*MAE*], [*IC ↑*], [*Dir.Acc.*],
    table.hline(stroke: 0.6pt + luma(120)),
    table.cell(rowspan: 5)[Кластерная],
    [LSTM],        [0.01314], [0.0812],   [*0.5138*],
    [GRU],         [0.01301], [*0.0958*], [0.5081],
    [CNN],         [0.01338], [0.0574],   [0.4934],
    [Transformer], [0.01321], [0.0689],   [0.5012],
    [LightGBM],    [0.01309], [0.0741],   [0.4978],
    table.hline(stroke: 0.6pt + luma(120)),
    table.cell(rowspan: 5)[Baseline],
    [LSTM],        [0.01401], [0.0289], [0.4821],
    [GRU],         [0.01408], [0.0518], [0.4811],
    [CNN],         [0.01389], [0.0048], [0.4681],
    [Transformer], [0.01394], [0.0171], [0.4751],
    [LightGBM],    [0.01381], [0.0341], [0.4791],
    table.hline(stroke: 1pt + black),
  ),
  caption: [Метрики — Handmade C3: кластерные vs baseline],
)

Анологично предыдущим кластерам, из-за большего процента примеров кризисного режима кластерные модели демонстрируют лучшее качество.

=== Промежуточные итоги и выводы

При нейтральных кластерах модели, обученные на всех данных, превосходят кластерные модели по метрикам в связи с большим количеством обучающих примеров.

Однако на кластерах с выраженным рыночным сигналом (бычий тренд, 
кризисный режим) кластерные модели превосходят baseline модели по 
метрикам качества.

Кластеризация на основе handmade-признаков обеспечивает в среднем более высокий прирост IC, чем кластеризация на основе TS2Vec-эмбеддингов благодаря важным для прогнозирования признакам, которые не удается извлечь из эмбеддингов.

#figure(
  table(
    columns: (1fr, 2fr, 1.4fr, 1fr, 1.4fr, 1fr),
    align: (center, left, center, center, center, center),
    inset: (x: 7pt, y: 5pt),
    stroke: none,
    table.hline(stroke: 1pt + black),
    [*Кластер*], [*Описание*], [*Лучший IC*], [*IC*], [*Лучший Dir.Acc*], [*Dir.Acc*],
    table.hline(stroke: 0.6pt + luma(120)),
    [C0], [Нейтральный],           [Baseline LSTM],  [0.0521], [Baseline LSTM],  [0.4961],
    [C1], [Бычий тренд],           [GRU],            [0.0891], [GRU],            [0.5221],
    [C2], [Высокая волатильность],  [LightGBM],       [0.0341], [LightGBM],       [0.4891],
    [C3], [Медвежий тренд],         [LSTM],           [0.0841], [GRU],            [0.5198],
    [C4], [Аномальные движения],    [GRU],            [0.0631], [Transformer],    [0.5147],
    table.hline(stroke: 1pt + black),
  ),
  caption: [Победители по кластерам — TS2Vec + KMeans],
)

#figure(
  table(
    columns: (1fr, 2fr, 1.4fr, 1fr, 1.4fr, 1fr),
    align: (center, left, center, center, center, center),
    inset: (x: 7pt, y: 5pt),
    stroke: none,
    table.hline(stroke: 1pt + black),
    [*Кластер*], [*Описание*], [*Лучший IC*], [*IC*], [*Лучший Dir.Acc*], [*Dir.Acc*],
    table.hline(stroke: 0.6pt + luma(120)),
    [C0], [Нейтральный],                    [Baseline LSTM], [0.0548], [Baseline GRU], [0.4948],
    [C1], [Бычий тренд],                     [GRU],           [0.0901], [Transformer],  [0.5214],
    [C2], [Нисходящий, отриц. асимметрия],   [LightGBM],      [0.0491], [LightGBM],     [0.4921],
    [C3], [Кризисный режим],                 [GRU],           [0.0958], [LSTM],         [0.5138],
    table.hline(stroke: 1pt + black),
  ),
  caption: [Победители по кластерам — Handmade + KMeans],
)





