#let bib_state = state("bib_state", bibliography("refs.bib", title: none))


// это штука, чтобы во вложенных файлах не было красных подчеркиваний. 
// в самих файлах нужно вставить

// #import "../state.typ": bib_state
// #context bib_state.get()
