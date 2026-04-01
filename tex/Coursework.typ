

// Настройки документа
#set document(
  title: "Прогнозирование цен акций на Московской бирже и автоматизированное формирование портфеля с торговым агентом",
  author: "Шадрин Андрей Романович",
)

// Настройки страницы
#set page(
  paper: "a4",
  margin: (
    left: 2.5cm,
    right: 1.0cm,
    top: 2.0cm,
    bottom: 2.0cm,
  ),
  numbering: "1",
)

// Настройки текста
#set text(
  font: "New Computer Modern",
  size: 12pt,
  lang: "ru",
)

// Междустрочный интервал 1.5
#set par(
  leading: 0.65em,
  first-line-indent: 1.25cm,
  justify: true,
)

// Настройки заголовков
#set heading(numbering: "1.1")

// Главы первого уровня жирным шрифтом
#show heading.where(level: 1): it => {
  set text(weight: "bold")
  it
}

// Настройки для рисунков и таблиц с нумерацией по секциям
#show figure.where(kind: image): set figure(numbering: num => {
  let h = counter(heading).get().first()
  numbering("1.1", h, num)
})

#show figure.where(kind: table): set figure(numbering: num => {
  let h = counter(heading).get().first()
  numbering("1.1", h, num)
})

// Настройки для уравнений с нумерацией по секциям
#set math.equation(numbering: num => {
  let h = counter(heading).get().first()
  numbering("(1.1)", h, num)
})

// Настройки для списков
#set enum(indent: 1em, numbering: "1.1.1.")

// Настройки для ссылок
#show link: set text(fill: blue)
#show cite: set text(fill: blue)

// Титульный лист
#include "title_kr.typ"
#pagebreak()

// Начинаем нумерацию со страницы 2
#counter(page).update(2)

// Содержание
#show outline.entry.where(level: 1): it => {
  set text(fill: black, weight: "bold")
  it
  v(0.65em)
}

#show outline.entry: it => {
  set text(fill: black)
  it
}

#outline(
  title: [Содержание],
  indent: auto,
)

#pagebreak()

#import "state.typ": bib_state
#bib_state.update(none)

// ========== АННОТАЦИЯ ==========
#{include "chapters/annotation.typ"}

// ========== ВВЕДЕНИЕ ==========
#{include "chapters/intro.typ"}
#{include "chapters/obzzzorr.typ"}
#{include "chapters/general_pipline.typ"}

// ========== ГЛАВА 1: ОПИСАНИЕ ИСПОЛЬЗУЕМЫХ АЛГОРИТМОВ ==========


#{include "chapters/data.typ"}
// ========== ГЛАВА 2: РЕАЛИЗОВАННЫЕ МОДЕЛИ И ПАЙПЛАЙН ==========
#{include "chapters/embeddings.typ"}
#{include "chapters/cluster.typ"}
#{include "chapters/chapter_models.typ"}
#{include "chapters/testing.typ"}
#{include "chapters/results.typ"}

// ========== ГЛАВА 3: ТРЕБОВАНИЯ К СИСТЕМЕ ==========

// #{include "chapters/chapter_func_and_no_func.typ"}

// ========== ГЛАВА 4: ПЛАН ДАЛЬНЕЙШЕЙ РАБОТЫ ==========

// #{include "chapters/chapter2.typ"}
// ========== БИБЛИОГРАФИЯ ==========






#bibliography("refs.bib", title: "Список литературы", style: "ieee")
