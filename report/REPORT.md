# Incertezza epistemica come guida all'adattamento di dominio source-free

### Replica ed estensione di Roy et al., *Uncertainty-guided Source-free Domain Adaptation* (U-SFAN, ECCV 2022) su riconoscimento di modalità locomotoria

**Corso:** Probabilistic Machine Learning — Università degli Studi di Trieste
**Codice:** `github.com/RiccardoSamaritan/BayesianExoAdaptation`

---

## Abstract

Un classificatore che stima un solo valore per ciascuno dei propri pesi, addestrato su una popolazione e applicato a un'altra, tende a sbagliare *con sicurezza*: le reti neurali con attivazioni lineari a tratti diventano tanto più confidenti quanto più si allontanano dai dati che hanno visto, e una procedura di adattamento non supervisionata che inseguisse ciecamente quelle predizioni può ribaltare il confine fra le classi nella direzione sbagliata. Questo progetto replica il meccanismo proposto da Roy et al. — approssimazione di Laplace esatta sull'ultimo strato, scomposizione dell'incertezza predittiva nelle sue componenti epistemica e aleatoria, e pesatura di ciascun dato target in base a quanto il modello si fida di sé stesso — e lo verifica in tre stadi di difficoltà crescente. Su un problema costruito in due dimensioni, dove il meccanismo è ispezionabile, la variante guidata dall'incertezza porta la classe spostata dal 6.0% al 98.7% di riconoscimento, dove la procedura non pesata resta ferma al 7.3%. Sul confronto fra cifre scritte a mano e immagini di abbigliamento, dove lo spostamento fra le due popolazioni è inequivocabile, la sola componente epistemica dell'incertezza raggiunge un'area sotto la curva di 0.987 come rilevatore, contro 0.946 dell'incertezza del modello non bayesiano. Sul riconoscimento di modalità locomotoria da sensore indossato (dataset UCI HAR), dove lo spostamento è naturale e mite perché dovuto alla sola variabilità fra persone, l'incertezza epistemica segue l'indicatore di spostamento con una correlazione di Spearman di 0.84 ± 0.03 su cinque inizializzazioni indipendenti (probabilità di accadere per caso inferiore all'1% in ciascuna), e il trattamento bayesiano migliora le misure di calibrazione su 7 soggetti su 11, con i guadagni concentrati sui più difficili. L'effetto dell'adattamento guidato è però positivo **soltanto** sul terzo di soggetti più spostati (+0.06 di accuratezza), con una dispersione fra inizializzazioni (±0.10) che comprende lo zero, e con un collasso parziale — almeno una classe che smette di essere predetta — nel 14–30% delle esecuzioni. Un'estensione a insieme aperto su dati di transizioni posturali produce un risultato *rovesciato* e stabile su cinque inizializzazioni: un controllo controfattuale isola la causa in un limite preciso e non ovvio del metodo, che rileva l'estrapolazione ma non la novità semantica interna alla regione dei dati noti. Il quadro complessivo è che il meccanismo funziona dove lo spostamento è forte, produce un effetto reale ma statisticamente debole dove è mite, e ha un limite riconosciuto e diagnosticato.

---

## 1. Nota metodologica sulla verificabilità

Ogni affermazione quantitativa in questo report è seguita dal riferimento al notebook e alla cella che l'ha prodotta, nella forma `(nome_notebook.ipynb, cella N)`, dove `N` è l'indice della cella nel file `.ipynb` (0-based, come nel JSON del notebook). Le figure **non** sono duplicate su disco: esistono come output inline nei notebook, e per ognuna è indicato dove trovarla. Dove un'interpretazione richiesta non è documentata nei notebook, è marcata `[DA DISCUTERE INSIEME]` invece di essere inferita.

**Convenzione di lettura.** Ogni metodologia impiegata nel progetto è accompagnata da un riquadro *Come funziona*, che la spiega in linguaggio corrente, senza sigle e senza dare per noti i termini inglesi. I riquadri servono a rendere il report leggibile senza consultare altro materiale, e possono essere saltati da chi conosce già il metodo: nessun risultato dipende dalla loro lettura. Le sigle e i termini inglesi privi di un equivalente italiano consolidato sono sciolti alla prima occorrenza e raccolti nel glossario dell'Appendice A.

Due parole ricorrono in tutto il testo e conviene fissarle subito. **Notebook** è un documento eseguibile che alterna testo, codice e risultati del codice (i file con estensione `.ipynb` nella cartella `notebooks/`): è il formato in cui tutti gli esperimenti di questo progetto sono stati svolti, e il motivo per cui ogni numero è verificabile senza rieseguire nulla. **Seed** (letteralmente «seme») è il numero da cui parte il generatore di numeri casuali: fissarlo rende un esperimento che usa il caso — l'inizializzazione dei pesi, il mescolamento dei dati — ripetibile identico, e ripeterlo con semi diversi permette di misurare quanto un risultato dipenda dalla fortuna anziché dal metodo.

I riferimenti alle dispense del corso (`PML_notes_full.pdf`) usano la numerazione di sezione citata in `README.md` e `TODO.md` §11; il PDF delle dispense non è versionato nel repository, quindi i numeri di sezione non sono verificabili dal repo stesso.

---

## 2. Introduzione e motivazione

### 2.1 Il problema

Nell'**adattamento di dominio senza accesso ai dati sorgente** (in inglese *source-free domain adaptation*) si dispone di un modello già addestrato su un insieme di dati etichettati — il **dominio sorgente** — e di dati **non** etichettati provenienti da un contesto diverso — il **dominio target** — ma non si ha più accesso ai dati sorgente originali. Allineare direttamente le due distribuzioni confrontandole è quindi impossibile: l'unica supervisione disponibile è il modello stesso.

> **Come funziona, in parole semplici.** «Dominio» significa qui la popolazione da cui i dati provengono: per esempio un gruppo di persone che indossano un sensore. Un modello addestrato su venti persone impara le regolarità di quelle venti; applicato a una ventunesima, che cammina in modo un po' diverso, può sbagliare. Questo cambiamento di popolazione fra addestramento e uso si chiama in inglese *domain shift*: nel seguito lo chiamiamo **spostamento di dominio**, o per brevità **shift**, termine che nel testo indica sempre e solo questo. La complicazione ulteriore è che i dati originali non sono più disponibili — per esempio perché contenevano informazioni personali che non si possono conservare, o perché sono troppo grandi da trasmettere — quindi non si può misurare la differenza fra le due popolazioni confrontandole direttamente: si può solo osservare come il modello si comporta sui dati nuovi.

Il fulcro del problema è che un modello **puntuale** — che per ogni suo peso stima un unico numero, senza dire quanto quel numero sia stato determinato dai dati — non ha alcuna nozione della propria ignoranza. Le reti neurali con attivazione ReLU (*rectified linear unit*, «unità lineare rettificata»: la funzione che lascia passare i valori positivi e azzera i negativi) estrapolano in modo lineare fuori dalla regione occupata dai dati di addestramento, e la loro uscita normalizzata — la funzione *softmax*, che converte i punteggi grezzi in probabilità che sommano a uno — diventa arbitrariamente vicina alla certezza anche in regioni mai viste (Hein et al., riferimento [23] del paper; Kristiadi, Hein & Hennig, ICML 2020, riferimento [26]).

> **Come funziona la patologia.** Una rete con attivazioni di questo tipo calcola, a grande distanza dai dati, una funzione che cresce linearmente. La probabilità assegnata alla classe vincente dipende dalla *differenza* fra i punteggi delle classi; se questa differenza cresce allontanandosi dai dati, la probabilità tende a uno. Il risultato è che più un dato è lontano da tutto ciò che la rete ha visto, più la rete si dichiara certa — l'opposto del comportamento desiderabile. È il fenomeno che in inglese si chiama *overconfidence*, cioè **eccesso di confidenza**.

Un obiettivo di adattamento non supervisionato come la **massimizzazione dell'informazione** (in inglese *information maximization*) — che premia le predizioni a bassa entropia, cioè quelle vicine alla certezza — amplifica questa patologia: dove il modello sorgente è sicuro e sbagliato, l'obiettivo rinforza l'errore invece di correggerlo.

### 2.2 Il paper replicato

Roy et al., *Uncertainty-guided Source-free Domain Adaptation* (U-SFAN, arXiv:2208.07591, ECCV 2022) propone una soluzione volutamente leggera. La tesi centrale, in tre passi:

1. Al modello sorgente si aggiunge un trattamento bayesiano del **solo ultimo strato** (in inglese *last-layer Laplace approximation*, §3.2 del paper, Eq. 5), lasciando deterministica la parte che estrae le caratteristiche dai dati. L'aggiunta è *a posteriori* (in latino *post hoc*: dopo che l'addestramento è finito, senza rifarlo) e costa un solo passaggio dei dati sorgente attraverso la rete.
2. Da quella distribuzione sui pesi si ricava, mediando su un campione di pesi plausibili (Eq. 6), una distribuzione di probabilità sulle predizioni, e da essa un peso per ciascun dato target, $w_i = \exp(-H_i)$, dove $H_i$ è l'entropia — cioè il grado di indecisione — della predizione media in quel punto.
3. Quel peso moltiplica **soltanto** il termine dell'obiettivo che spinge verso predizioni sicure (Eq. 7): i dati target su cui il modello ammette di non sapere influiscono meno sull'aggiornamento. Il termine che impone varietà fra le classi resta invece non pesato.

> **Come funziona l'idea, in una frase.** Il modello, prima di adattarsi, dichiara per ciascun dato nuovo quanto si fida di sé stesso; l'adattamento poi ascolta molto i dati su cui il modello è informato e poco quelli su cui sta tirando a indovinare. Senza questa graduazione, l'adattamento tratta le proprie ipotesi peggiori come se fossero verità e vi si adegua.

Il vantaggio rivendicato non è tanto l'accuratezza media, quanto la **robustezza a spostamenti di dominio forti** — precisamente il regime in cui il metodo convenzionale fallisce (Fig. 4b del paper: «IM, when used with a MAP estimate, finds a completely flipped decision boundary», cioè: la massimizzazione dell'informazione applicata a una stima puntuale trova un confine di decisione completamente ribaltato).

### 2.3 Perché è un progetto di Probabilistic Machine Learning

Il progetto non usa il vocabolario bayesiano come decorazione: ogni suo passaggio è un oggetto visto a lezione.

- Il trattamento bayesiano dell'ultimo strato **è** regressione logistica bayesiana multiclasse con approssimazione di Laplace (dispense §7.3–7.4), applicata su feature apprese $\phi(x) = g_\beta(x)$ invece che scelte a mano. L'Hessiana implementata è la generalizzazione a $K$ classi della formula delle dispense, e la riduzione a $K=2$ è verificata numericamente (§5.1 di questo report).
- La decomposizione dell'incertezza è **esattamente** l'informazione mutua fra predizione e parametri (dispense §2.3), stessa forma funzionale che compare — su un oggetto diverso — nell'obiettivo di adattamento (§4.4).
- La predizione è una **media bayesiana fra modelli** (in inglese *Bayesian model averaging*, dispense §10.6): non una singola predizione, ma la media su un ventaglio di ipotesi, ciascuna pesata da quanto è plausibile alla luce dei dati.
- Il campionamento con **catene di Markov Monte Carlo** (in inglese *Markov chain Monte Carlo*, dispense cap. 8) è usato non come metodo di inferenza, ma come **verità di riferimento** per validare l'approssimazione, su un problema abbastanza piccolo perché il confronto sia possibile.

La domanda sperimentale è quindi una domanda probabilistica: *l'incertezza sui parametri stimata da un'approssimazione locale sa effettivamente di cosa parla, e quella conoscenza è utile?*

---

## 3. Background teorico

### 3.1 Approssimazione di Laplace e regressione logistica bayesiana (dispense §7.3–7.4)

> **Due termini che ricorrono in tutto il capitolo.** La **distribuzione a priori** (in inglese *prior*) è ciò che si assume sui pesi *prima* di guardare i dati: qui, che valori grandi siano improbabili. La **distribuzione a posteriori** (in inglese *posterior*) è ciò che si crede sui pesi *dopo* aver visto i dati, cioè la combinazione fra l'assunzione iniziale e l'evidenza portata dalle osservazioni. Tutto il progetto ruota attorno al fatto che un addestramento ordinario restituisce un solo punto di questa seconda distribuzione — il suo massimo — mentre l'informazione sull'incertezza sta nella sua *forma*.

Sia $f = h \circ g$, dove $g$ è l'estrattore di caratteristiche (parametri $\beta$) e $h$ la testa lineare (parametri $\theta$). Fissato $\beta$ al valore trovato in addestramento, la funzione da massimizzare rispetto alla sola testa è

$$\log p(\theta \mid \mathcal{D}) = \sum_n \log \varphi_{y_n}\!\big(\theta^\top \phi_n\big) - \tfrac{1}{2}\tau \lVert\theta\rVert^2 + \text{cost.}, \qquad \phi_n = [\,g_{\beta_{\text{MAP}}}(x_n);\,1\,]$$

cioè **letteralmente** la funzione obiettivo di una regressione logistica multiclasse con distribuzione a priori gaussiana isotropa di precisione $\tau$, con la sola differenza che le caratteristiche $\phi_n$ sono apprese anziché progettate a mano.

> **Come funziona l'approssimazione di Laplace.** Dopo l'addestramento si conosce un solo punto: la combinazione di pesi che meglio spiega i dati, chiamata *stima a massima probabilità a posteriori* (in inglese *maximum a posteriori*, abbreviato MAP; nel seguito «stima puntuale» o «soluzione MAP»). L'idea è di guardare non solo l'altezza di quel massimo, ma la **forma della collina attorno a esso**. Se lungo una certa direzione la funzione obiettivo cala ripidamente appena ci si allontana dal massimo, allora quella direzione è ben vincolata dai dati: cambiare i pesi in quel senso peggiora subito l'adattamento, quindi il peso è determinato con precisione. Se invece la funzione è quasi piatta lungo un'altra direzione, i pesi potevano assumere molti valori diversi senza che i dati se ne accorgessero: quella libertà residua **è** l'incertezza. La curvatura in tutte le direzioni è raccolta in una matrice di derivate seconde, l'**Hessiana**; il suo inverso descrive quanto i pesi possono variare, e si usa come matrice di covarianza di una gaussiana centrata sulla stima puntuale. In sintesi: curvatura alta → poca incertezza; curvatura bassa → molta incertezza.

> **Che ruolo ha la distribuzione a priori.** Il termine $\tau\lVert\theta\rVert^2$ dice, prima di vedere i dati, che pesi molto grandi sono improbabili. Il numero $\tau$ si chiama *precisione*: più è alto, più la nostra convinzione iniziale è stretta attorno allo zero, e più occorrono dati per allontanarsene. Nella pratica dell'ottimizzazione lo stesso termine è noto come **decadimento dei pesi** (in inglese *weight decay*); §4.5 spiega come i due numeri si corrispondono.

Formalmente, l'approssimazione di Laplace è lo sviluppo di Taylor al secondo ordine attorno al massimo:

$$p(\theta \mid \mathcal{D}) \approx \mathcal{N}\!\big(\theta \mid \theta_{\text{MAP}}, H^{-1}\big), \qquad H = -\nabla^2_\theta \log p(\theta\mid\mathcal{D})\big|_{\theta_{\text{MAP}}}$$

Per la log-verosimiglianza softmax l'Hessiana esatta (Gauss–Newton generalizzata) è

$$H = \sum_{n=1}^{N} \Lambda_n \otimes \phi_n \phi_n^\top + \tau I, \qquad \Lambda_n = \operatorname{diag}(p_n) - p_n p_n^\top$$

con $p_n$ il vettore di probabilità prodotto dalla stima puntuale nel punto $n$, $K$ il numero di classi e $\otimes$ il prodotto di Kronecker (l'operazione che costruisce una matrice grande combinando ogni elemento della prima con tutta la seconda).

> **Come si legge questa formula.** Ogni dato di addestramento contribuisce un pezzetto di curvatura, e i pezzetti si sommano: più dati, più curvatura, meno incertezza — esattamente il comportamento che ci si aspetta. Il contributo di un singolo dato è il prodotto di due fattori. Il primo, $\Lambda_n$, dipende da quanto la predizione in quel punto è indecisa: vale quasi zero se il modello è già certissimo (un dato su cui non c'è niente da imparare) ed è massimo quando le classi sono equiprobabili. Il secondo, $\phi_n\phi_n^\top$, dice *in quali direzioni* dello spazio delle caratteristiche quel dato porta informazione. Il termine finale $\tau I$ è il contributo della distribuzione a priori, che aggiunge curvatura in tutte le direzioni allo stesso modo e garantisce che la matrice sia invertibile anche dove i dati non dicono nulla.

Nel caso $K=2$ si ha $\Lambda_n = s_n(1-s_n)\begin{bmatrix}1 & -1\\ -1 & 1\end{bmatrix}$, e ogni blocco diagonale di $H$ si riduce a $S_N^{-1} = S_0^{-1} + \sum_n s_n(1-s_n)\phi_n\phi_n^\top$, la formula binaria delle dispense §7.4. Questa riduzione è verificata numericamente (§5.1).

L'implementazione è in `src/bayesian.py` (`LastLayerLaplace.fit`), completamente vettorizzata via `einsum`, con simmetrizzazione `cov = 0.5*(cov + cov.T)` dopo l'inversione per rimuovere l'asimmetria numerica.

Il carattere **locale e unimodale** dell'approssimazione è una proprietà da tenere presente, non un dettaglio: il metodo guarda la curvatura attorno a *un solo* massimo e non sa nulla del resto della distribuzione (Fig. 2a del paper). «Unimodale» significa che la gaussiana usata come approssimazione ha un unico picco, mentre la distribuzione vera potrebbe averne più di uno, o essere asimmetrica: in quel caso l'approssimazione descrive bene la zona vicino al picco e male tutto il resto. Il §8.2 mostra, sui dati di questo progetto, dove precisamente questo si vede.

### 3.2 Entropia, divergenza di Kullback–Leibler, informazione mutua (dispense §2.3): la decomposizione dell'incertezza e l'obiettivo di adattamento hanno la stessa forma

> **Come funzionano le tre quantità in gioco.** L'**entropia** misura quanta indecisione contiene una distribuzione di probabilità: è zero se una classe ha probabilità uno e tutte le altre zero (nessun dubbio), ed è massima quando tutte le classi sono equiprobabili (dubbio totale). Si misura in *nat*, l'unità che si ottiene usando il logaritmo naturale; con tre classi il valore massimo possibile è $\log 3 \approx 1.10$ nat, e questo tetto serve da riferimento per giudicare se un'entropia osservata è grande o piccola. La **divergenza di Kullback–Leibler** misura quanto una distribuzione è diversa da un'altra presa come riferimento: vale zero se coincidono e cresce al divergere. L'**informazione mutua** fra due grandezze misura quanto conoscere l'una riduce l'incertezza sull'altra: vale zero se sono indipendenti.

> **Come si combinano nella decomposizione dell'incertezza.** Si immagini di avere non un modello ma una commissione di modelli, tutti compatibili con i dati di addestramento. Per un dato nuovo si raccolgono i voti. Se i membri della commissione sono tutti d'accordo ma ciascuno è indeciso (per esempio tutti dicono «metà probabilità classe A, metà classe B»), l'incertezza è **intrinseca al dato**: nessun membro potrebbe fare meglio, perché le due classi si sovrappongono davvero là. Questa componente si chiama **aleatoria**, e non si riduce raccogliendo più dati di addestramento. Se invece ciascun membro è sicuro di sé ma i membri si contraddicono a vicenda (uno dice A con certezza, un altro B con certezza), l'incertezza sta **nel modello**: i dati non sono bastati a decidere quale membro abbia ragione. Questa componente si chiama **epistemica**, e si ridurrebbe con più dati. L'informazione mutua fra la predizione e i pesi misura esattamente il secondo caso: quanto la predizione cambia al variare dell'ipotesi. Questa costruzione è nota in letteratura con la sigla BALD (*Bayesian Active Learning by Disagreement*, «apprendimento attivo bayesiano per disaccordo»), dal contesto in cui fu introdotta; nel seguito la chiamiamo semplicemente **decomposizione epistemica/aleatoria**.

Formalmente, data una distribuzione predittiva ottenuta mediando su un insieme di ipotesi, la decomposizione è

$$\underbrace{\mathbb{H}\!\left[\mathbb{E}_{\theta}\, p(y\mid x,\theta)\right]}_{\text{totale}} \;=\; \underbrace{\mathbb{E}_{\theta}\!\left[\mathbb{H}\!\left[p(y\mid x,\theta)\right]\right]}_{\text{aleatoria}} \;+\; \underbrace{\mathbb{I}\!\left[y;\theta \mid x\right]}_{\text{epistemica (BALD)}}$$

L'epistemica è dunque l'**informazione mutua** fra la predizione e i parametri: è grande quando ipotesi diverse, tutte plausibili sotto la posterior, sono in disaccordo su quel punto. È l'incertezza che si ridurrebbe con più dati. L'aleatoria è ciò che resterebbe anche conoscendo esattamente i pesi: sovrapposizione intrinseca fra classi.

L'obiettivo di adattamento ha **esattamente la stessa forma su un oggetto diverso**. La funzione obiettivo della massimizzazione dell'informazione (paper Eq. 2–3, implementata in `src/im_adapt.py::im_loss`) è

$$\mathcal{L}_{\text{IM}}^{\text{(da massimizzare)}} = \underbrace{\mathbb{H}\!\left[\tfrac{1}{N}\textstyle\sum_i p(y\mid x_i)\right]}_{\text{entropia della media}} - \underbrace{\tfrac{1}{N}\textstyle\sum_i \mathbb{H}\!\left[p(y\mid x_i)\right]}_{\text{media delle entropie}} \;=\; \mathbb{I}\!\left[y;x\right]_{\hat{p}_{\text{emp}}}$$

cioè l'informazione mutua fra il dato e la predizione, calcolata sulla distribuzione empirica dei dati target. La media è **sui dati** invece che **sui pesi**. Il termine di diversità del paper, $\mathcal{L}_{\text{div}} = D_{\mathrm{KL}}(\hat p \,\|\, K^{-1}\mathbf{1}) - \log K = -\mathbb{H}[\hat p]$, è la stessa entropia della media riscritta come divergenza di Kullback–Leibler dalla distribuzione uniforme (identità verificata nel commento di documentazione di `src/im_adapt.py`).

> **Come funziona questo obiettivo.** Ha due spinte contrapposte, e servono entrambe. La prima chiede che **ogni singola predizione** sia netta, cioè quasi tutta su una classe: è il termine «media delle entropie», che va reso piccolo. Da sola, questa spinta ha una scorciatoia disastrosa: assegnare *tutti* i dati alla stessa classe soddisfa perfettamente la richiesta di predizioni nette. La seconda spinta impedisce la scorciatoia, chiedendo che **l'insieme delle predizioni** sia vario, cioè che le classi siano tutte rappresentate: è il termine «entropia della media», che va reso grande. Il parametro $\gamma$ decide quanto pesa la seconda rispetto alla prima. Il fenomeno per cui la prima spinta, lasciata sola, manda tutte le predizioni su un'unica classe è ciò che nel report chiamiamo **collasso**, e nell'esperimento di §6.5 è verificato come controllo negativo.

Questa coincidenza di forma è, didatticamente, il punto più elegante del progetto: la stessa quantità informazionale serve, su un fronte, a *misurare* quanto il modello non sa (mediando sui pesi) e, sull'altro, a *ottimizzare* la struttura delle predizioni sul target (mediando sui dati).

Un'avvertenza di implementazione, facile da confondere: il peso che entra nella funzione obiettivo è $w_i = \exp(-H_i^{\text{totale}})$, cioè si basa sull'entropia **totale**, non sulla sola componente epistemica (paper Eq. 7). La forma esponenziale con segno negativo fa sì che il peso valga uno quando l'incertezza è nulla e tenda a zero al crescere dell'incertezza, senza mai diventare negativo. La decomposizione epistemica/aleatoria serve come *diagnostica* del perché $H_i$ sia grande; la variante che pesa con la sola componente epistemica è una deviazione deliberata, provata come variante (f) nello studio comparativo di §6.5.

### 3.3 Media bayesiana fra modelli (dispense §10.6)

La predizione non è quella del singolo modello puntuale, $\varphi(h_{\theta_{\text{MAP}}}(z))$, ma la media su tutte le ipotesi plausibili (paper Eq. 5–6), calcolata approssimativamente per campionamento:

$$p(y_k \mid z, \mathcal{D}) \approx \frac{1}{M}\sum_{j=1}^{M} \varphi_k\!\big(h_{\theta_j}(z)/\tau\big), \qquad \theta_j \sim \mathcal{N}(\theta_{\text{MAP}}, H^{-1})$$

> **Come funziona, in parole semplici.** La media che ci interessa è un integrale su tutti i possibili valori dei pesi, pesati per la loro plausibilità: non si può calcolare in forma chiusa. Il metodo **Monte Carlo** lo sostituisce con una media su un numero finito di estrazioni casuali: si estraggono $M$ combinazioni di pesi dalla gaussiana trovata al passo precedente, si calcola la predizione di ciascuna, e si fa la media delle predizioni. Più estrazioni si fanno, più la media si avvicina all'integrale vero — con un errore che cala come la radice quadrata del numero di estrazioni, motivo per cui $M$ deve essere scelto e non indovinato (§6.2). Il nome «Monte Carlo» viene dal casinò: si risolve un problema di calcolo tirando ripetutamente a sorte.

> **Come si estraggono pesi correlati.** Estrarre da una gaussiana multivariata non è come estrarre numeri indipendenti: le direzioni dello spazio dei pesi sono correlate fra loro, e le correlazioni sono descritte dalla matrice di covarianza. La **decomposizione di Cholesky** scompone quella matrice in un fattore triangolare; moltiplicando per quel fattore un vettore di numeri casuali indipendenti si ottengono estrazioni che rispettano esattamente le correlazioni richieste. È il modo standard, e numericamente stabile, di fare questa operazione.

Il campionamento avviene tramite fattore di Cholesky della covarianza (`LastLayerLaplace.sample_heads`). Il parametro $\tau$, chiamato **temperatura** nel codice (`temperature`), divide i punteggi grezzi prima della normalizzazione softmax, per ciascuna estrazione: valori sotto uno rendono ogni singola predizione più netta, valori sopra uno la appiattiscono verso l'uniforme. Il numero $M$ di estrazioni non è un dettaglio implementativo ma un iperparametro con un criterio di scelta esplicito (§6.2).

### 3.4 Il campionamento a catene di Markov come verità di riferimento (dispense cap. 8)

L'approssimazione di Laplace è, per costruzione, un'approssimazione: prima di fidarsene conviene confrontarla con la distribuzione vera, su un problema abbastanza piccolo perché quest'ultima sia ottenibile. Il notebook `test_laplace_vs_mcmc.ipynb` fa esattamente questo, campionando la distribuzione esatta dei pesi della testa (12 parametri) con l'algoritmo di **Metropolis–Hastings**.

> **Come funziona Metropolis–Hastings.** Non si sa come estrarre direttamente dalla distribuzione vera, ma si sa come *valutarla* in un punto qualsiasi, a meno di una costante. L'algoritmo sfrutta questo: parte da una combinazione di pesi, ne propone una vicina scelta a caso (una **passeggiata aleatoria**, in inglese *random walk*), e decide se accettarla confrontando le due plausibilità. Se la proposta è più plausibile, la accetta sempre; se è meno plausibile, la accetta comunque con una probabilità pari al rapporto fra le due. Ripetendo molte volte, la sequenza di punti visitati — la **catena** — passa nelle varie regioni dello spazio dei pesi in proporzione alla loro plausibilità, e l'insieme dei punti visitati diventa un campione della distribuzione vera. Due dettagli pratici: i primi punti dipendono ancora da dove si è partiti e vanno scartati (fase di **rodaggio**, in inglese *burn-in*); e il **tasso di accettazione**, cioè la frazione di proposte accettate, indica se il passo è ben tarato — troppo grande e quasi tutto viene rifiutato, troppo piccolo e la catena si muove con esasperante lentezza. Valori intorno a un quarto o un terzo sono considerati buoni.

Questo è il collaudo dello strumento, non un metodo alternativo di inferenza: nella catena di elaborazione reale la distribuzione sui pesi va interrogata migliaia di volte durante l'adattamento, e un campionamento di questo tipo sarebbe proibitivamente lento. I risultati, incluse le discrepanze, sono in §5.2 e §8.2.

---

## 4. Metodologia

### 4.1 Architettura e dimensione dell'Hessiana

Rete neurale completamente connessa (in inglese *multi-layer perceptron*, «percettrone multistrato») con dimensioni `561 → 128 → 64 → 3`: l'estrattore $g$ è composto da due strati lineari con attivazione ReLU, la testa $h$ è un singolo strato lineare da 64 ingressi a 3 uscite.

> **Come è divisa la rete, e perché.** I 561 numeri che descrivono una finestra di segnale entrano nella rete e vengono trasformati due volte, fino a diventare 64 numeri: questa parte si chiama **estrattore di caratteristiche** (in inglese *feature extractor*), e il suo compito è costruire una rappresentazione in cui le tre attività siano facilmente distinguibili. L'ultimo strato, la **testa** (in inglese *head*), prende quei 64 numeri e produce tre punteggi, uno per classe. La separazione conta perché il trattamento bayesiano si applica **soltanto alla testa**: si mantiene una sola risposta per l'estrattore e un ventaglio di risposte plausibili per la testa. Il motivo è pratico e insieme concettuale — l'ultimo strato è piccolo, quindi il calcolo è esatto e veloce, e quello strato è anche dove la decisione fra le classi viene effettivamente presa.

Conteggi verificati: 80 387 parametri totali, di cui 80 192 nell'estrattore e **195 nella testa** (`har_source_training.ipynb`, cella 13). La posterior è quindi su $K\cdot(D+1) = 3 \times 65 = 195$ parametri, e $H$ è una matrice $195 \times 195$: covarianza confermata definita positiva, autovalori in $[3.10\times10^{-4},\, 3.98\times10^{-2}]$, simmetria esatta (`max |cov - cov^T| = 0.00e+00`) (`har_source_training.ipynb`, cella 23).

**Perché la sezione §6 del piano di lavoro — l'approssimazione fattorizzata dell'Hessiana — è stata deliberatamente saltata.** Il paper originale ricorre a un'approssimazione chiamata *Kronecker-factored Laplace approximation* (approssimazione di Laplace fattorizzata secondo Kronecker, abbreviata in letteratura con la sigla KFAC) perché lavora con reti molto grandi, del tipo ResNet-50, e con molte classi: in quel caso anche la sola Hessiana dell'ultimo strato può diventare ingestibile.

> **Come funziona quell'approssimazione, e perché qui non serve.** L'Hessiana esatta è una somma di tanti pezzi, uno per dato, e ciascun pezzo è il prodotto di due matrici più piccole. Se la matrice complessiva fosse essa stessa il prodotto di due matrici piccole, si potrebbe invertirla invertendo separatamente i due fattori — molto più economico. Il problema è che una *somma* di prodotti non è, in generale, un prodotto: l'uguaglianza vale solo se si assume che i due fattori siano fra loro indipendenti, che è appunto un'approssimazione. Nel nostro caso la matrice da invertire ha dimensione $195 \times 195$ e si inverte esattamente in millisecondi: introdurre quell'ipotesi di indipendenza darebbe un risultato **peggiore** di una soluzione esatta già disponibile, senza alcun risparmio di calcolo apprezzabile a questa scala. La scelta è documentata nel docstring di `src/bayesian.py` («This is exact (no KFAC) because the feature dimension keeps K·(D+1) small enough to invert directly») e in `TODO.md` §6. È un'omissione motivata, non un item incompleto: la conseguenza è che ogni risultato di questo report usa l'Hessiana **esatta**, non un'approssimazione di essa.

### 4.2 I dati, e il cambio di dataset a metà progetto

Il progetto nasceva su un insieme di dati di **elettromiografia di superficie** dell'arto inferiore (in inglese *surface electromyography*, cioè la misura dell'attività elettrica dei muscoli tramite elettrodi appoggiati sulla pelle), denominato BASAN, con l'idea di addestrare su soggetti sani e adattare a soggetti con una patologia al ginocchio (`docs/project_overview_it.md`). Quei dati sono stati accantonati per problemi di qualità del segnale grezzo, la cui soluzione avrebbe richiesto un lavoro di elaborazione del segnale sproporzionato rispetto all'obiettivo metodologico del corso; l'implementazione attuale usa **UCI HAR** (*Human Activity Recognition*, «riconoscimento dell'attività umana»): 30 soggetti, uno smartphone fissato alla vita, e per ogni finestra temporale di 2.56 secondi un vettore di 561 caratteristiche già calcolate dagli autori del dataset (medie, deviazioni standard, energie in banda e simili, estratte da accelerometro e giroscopio). Il framework bayesiano, la domanda di ricerca e il piano di validazione sono rimasti quelli originali. Il dettaglio e le conseguenze sono in §8.3.

Validazione del caricamento: 10 299 finestre × 561 caratteristiche, 30 soggetti, 0 NaN, 0 Inf, tutti i valori in $[-1,1]$, 0 valori fuori range (`loader.ipynb`, cella 6). Il task principale è la locomozione a 3 classi (WALKING / WALKING_UPSTAIRS / WALKING_DOWNSTAIRS), 4 672 finestre (`loader.ipynb`, cella 11).

### 4.3 Suddivisione per soggetto e indicatore di spostamento

La suddivisione in addestramento e verifica fornita col dataset è scartata; la partizione è fatta **per soggetto**, con seme 42: un insieme sorgente di 20 soggetti (`1,2,3,4,5,6,9,10,12,13,14,16,17,18,23,24,25,26,28,29`, 3 197 finestre) e 10 soggetti target (`7,8,11,15,19,20,21,22,27,30`, 1 475 finestre) (`loader.ipynb`, cella 11).

> **Perché la divisione è per soggetto e non per finestra.** Le finestre temporali dello stesso soggetto sono molto simili fra loro. Se si mescolassero e poi si dividessero a caso, finestre quasi identiche finirebbero sia in addestramento sia in verifica, e il modello sembrerebbe bravissimo semplicemente perché sta riconoscendo persone già viste. Separando i soggetti si costringe il modello a generalizzare a una persona nuova, che è il problema che interessa davvero. È lo stesso motivo per cui, dentro l'insieme sorgente, la verifica interna usa 4 soggetti tenuti completamente da parte (§6.1).

> **Come funziona la standardizzazione, e perché solo sulla sorgente.** Le 561 caratteristiche hanno scale diverse fra loro; per renderle confrontabili a ciascuna si sottrae la propria media e si divide per la propria deviazione standard, così che ognuna abbia media zero e dispersione uno. Media e deviazione standard sono calcolate **soltanto** sui dati sorgente e poi applicate invariate ai target: se si ricalcolassero sui target si userebbe informazione sul dominio di arrivo, che nell'impostazione senza accesso ai dati sorgente non è lecito assumere di avere, e si nasconderebbe parte dello spostamento di dominio proprio nella normalizzazione.

Serve poi una misura, per ciascun soggetto target, di **quanto** sia lontano dalla sorgente. La misura scelta — chiamata nel report *indicatore di spostamento*, cioè indicatore indiretto dello spostamento — è la distanza euclidea fra il centroide del soggetto e la media dell'insieme sorgente, nello spazio standardizzato a 561 dimensioni (`loader.ipynb`, cella 13):

> **Come funziona questo indicatore.** Il **centroide** di un soggetto è il punto medio di tutte le sue finestre: un unico vettore di 561 numeri che ne riassume il comportamento. Si calcola quindi la distanza in linea d'aria fra quel punto e il punto medio di tutta la sorgente: più è grande, più quel soggetto «assomiglia poco» alla popolazione di addestramento. È una misura volutamente grezza — riassume una nuvola di punti con il suo solo centro, ignorandone la forma e la dispersione — e serve come riferimento **indipendente dal modello**: siccome non usa né i pesi né le predizioni, correlarla con l'incertezza del modello non è circolare.

| soggetto | 27 | 15 | 21 | 7 | 30 | 11 | 22 | 20 | 8 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|
| distanza | 7.41 | 9.67 | 10.03 | 10.16 | 11.06 | 11.41 | 18.19 | 18.43 | 19.89 | 29.21 |

Media 14.54, deviazione standard 6.39.

> **Come si stabilisce se una distanza è grande.** Un numero come 14.54 non significa niente da solo: serve un termine di paragone che rappresenti l'assenza di spostamento. Si costruisce così: si prende un soggetto **della sorgente**, si finge che sia un soggetto nuovo, e si misura la distanza fra il suo centroide e la media degli altri diciannove; si ripete per tutti e venti e si fa la media. Questa procedura si chiama *leave-one-subject-out*, cioè «lascia fuori un soggetto per volta». Il valore che ne risulta è la distanza tipica fra due soggetti che appartengono alla **stessa** popolazione: se le distanze dei target sono paragonabili, lo spostamento di dominio non è maggiore della normale variabilità fra persone. Un secondo controllo, ancora più elementare, divide a caso le finestre della sorgente in due metà ignorando chi le ha prodotte: lì lo spostamento è per costruzione nullo, quindi la distanza misurata dice quanto vale il puro rumore di misura.

**L'indicatore va preso per quello che è**: il riferimento a spostamento quasi nullo — distanze *lascia-fuori-un-soggetto* calcolate **dentro** l'insieme sorgente — vale in media 12.15 ± 5.00, quindi il rapporto target/sorgente è solo **1.20×**, sotto la soglia di 2× che il notebook stesso si era dato; il notebook stampa infatti «⚠ Target shifts are comparable to source baseline - minimal shift detected» (`loader.ipynb`, cella 15). Il controllo di sanità (split casuale di finestre ignorando il soggetto) dà 0.95, cioè praticamente zero, confermando che il riferimento è costruito bene. La conclusione onesta è che **su HAR lo shift medio è mite**; ciò che il resto del progetto sfrutta è la sua forte eterogeneità (7.41 a 29.21), non la sua entità media. Questo è determinante per interpretare l'ablazione (§6.5).

> **Come funziona la visualizzazione usata per il controllo grafico.** Non si possono disegnare 561 dimensioni. L'**analisi delle componenti principali** (in inglese *principal component analysis*) cerca le due direzioni lungo le quali i dati si disperdono di più e proietta tutto su quel piano: si ottiene un disegno bidimensionale che conserva la maggior parte della struttura, ma non tutta. Nel nostro caso le due direzioni scelte spiegano il 28.6% e il 9.0% della dispersione totale, cioè il 37.65% complessivo: abbastanza per un controllo visivo di plausibilità, non abbastanza per trarne conclusioni quantitative — che è esattamente l'uso che ne viene fatto qui.

*Figura: `notebooks/loader.ipynb`, cella 19 — sezione "9. Visualization: Source vs Target in PCA Space" (due pannelli: tutte le finestre sorgente contro target, e centroidi per soggetto etichettati). Proiezione calcolata solo sulla sorgente, dispersione spiegata 28.6% + 9.0% = 37.65% (cella 17).*

### 4.4 Protocollo di adattamento

Adattamento senza dati sorgente, su dati target non etichettati: si aggiorna **solo** $\beta$, cioè l'estrattore di caratteristiche $g$, mentre la testa $h$ resta congelata assieme alla propria distribuzione sui pesi (paper §3, Fig. 3b; `src/im_adapt.py::adapt_target`, che disattiva il calcolo del gradiente sui parametri della testa).

> **Come funziona un passo di adattamento.** Il ciclo, ripetuto 300 volte, fa quattro cose in quest'ordine. (1) Fa passare tutti i dati target attraverso la rete nella sua configurazione attuale, ottenendo una predizione per ciascuno. (2) Interroga la testa bayesiana per stimare quanta incertezza c'è su ciascun dato, e ne ricava un peso fra zero e uno. (3) Calcola la funzione obiettivo descritta in §3.2, con il termine «predizioni nette» moltiplicato per quei pesi. (4) Modifica leggermente i soli parametri dell'estrattore nella direzione che migliora l'obiettivo. «Congelare la testa» significa che il passo (4) non la tocca: la regola di decisione finale resta quella imparata sulla sorgente, e ciò che si adatta è la rappresentazione che le viene data in ingresso. Questo è il senso dell'espressione inglese *source hypothesis transfer*, «trasferimento dell'ipotesi sorgente»: si conserva l'ipotesi e si sposta il dato.

> **Perché i pesi sono ricalcolati a ogni passo.** Man mano che l'estrattore cambia, le caratteristiche di ciascun dato cambiano, e con esse l'incertezza del modello su quel dato. Congelare i pesi calcolati al primo passo significherebbe usare per 300 passi un giudizio di affidabilità che si riferisce a una rete che non esiste più. Ricalcolarli costa un campionamento Monte Carlo per passo, ed è il motivo per cui $M$ durante l'adattamento è tenuto a 100 e non a 5000.

Gli iperparametri sono fissati nel piano di lavoro (`TODO.md` §1) e usati identici in tutte le varianti dello studio comparativo: $\gamma = 0.5$ (il bilanciamento fra i due termini), $\tau = 0.4$ (la temperatura), 300 passi, ottimizzatore Adam con passo di apprendimento $10^{-2}$, e $M = 100$ estrazioni di pesi per passo.

La riduzione di peso dei dati incerti tocca solo il termine che spinge verso predizioni nette, mai il termine di diversità $\mathcal{L}_{\text{div}}$ (verificato in `im_loss`: `ent_term = (weights * per_sample_ent).mean()`, mentre `div_term = -entropy(p_hat)` non vede i pesi).

Due dettagli implementativi che vale enunciare per precisione: (i) i punteggi grezzi che entrano nella funzione obiettivo sono divisi per $\tau = 0.4$, mentre la distribuzione predittiva usata per calcolare i pesi è invocata con la temperatura lasciata al valore predefinito 1.0 — le due temperature quindi non coincidono; (ii) la chiamata alla predittiva dentro il ciclo non riceve un generatore di numeri casuali dedicato, quindi il rumore di campionamento cambia da un passo all'altro (l'esperimento nel suo insieme resta ripetibile perché il seme globale è fissato all'inizio). Il secondo punto contribuisce alla rumorosità visibile nell'andamento dei pesi.

**Nessuna etichetta dei soggetti target è mai usata per l'adattamento o per scegliere il modello.** Le etichette target compaiono esclusivamente nel calcolo delle metriche finali riportate, cioè solo per *misurare* a posteriori quanto bene le cose sono andate. Lo stesso vale per la scelta di $M$, di $\tau$, del numero di passi e dei semi: tutti fissati in anticipo o scelti su dati sorgente. Questa disciplina è ciò che distingue un esperimento di adattamento non supervisionato da un esperimento che, senza dichiararlo, sbircia la risposta.

### 4.5 Convenzione su $\tau_{\text{prior}}$ e le sue due validazioni indipendenti

Il piano di lavoro (`TODO.md` §1b) impone $\tau_{\text{prior}} = \text{decadimento dei pesi} \times N$. Nella catena di elaborazione su UCI HAR: $0.01 \times 2510 = 25.10$ (`har_source_training.ipynb`, cella 15).

> **Perché quella moltiplicazione.** Durante l'addestramento la penalizzazione sui pesi grandi è applicata alla funzione di costo **media** sui dati; nella formula dell'Hessiana, invece, i contributi dei dati sono **sommati**. Perché la penalizzazione usata in addestramento e la distribuzione a priori usata nel calcolo dell'incertezza descrivano la stessa convinzione iniziale, il coefficiente medio va moltiplicato per il numero di dati. Sbagliare questa conversione — usare il coefficiente così com'è, cioè 0.01 invece di 25.10 — significherebbe assumere una convinzione iniziale mille volte più debole di quella effettivamente usata in addestramento, e produrrebbe incertezze sistematicamente troppo grandi. È il motivo per cui il piano di lavoro la fissa esplicitamente come requisito.

**Nota di precisione onesta:** il fattore usato è $N_{\text{train}} = 2510$ (il sottoinsieme di addestramento dell'insieme sorgente), mentre l'Hessiana è poi sommata su tutte le 3 197 finestre dell'insieme sorgente completo, e `TODO.md` §1b parla di $N_{\text{source}}$. La discrepanza nasce dal fatto che `tau_prior` è restituito da `train_source_model` (che conosce solo il suo training set) mentre `LastLayerLaplace.fit` è chiamato sul pool intero. Con $\tau = 25.10$ invece di $31.97$ la prior è leggermente più debole del previsto; l'effetto sui risultati è presumibilmente marginale, ma la convenzione dichiarata e quella implementata non sono identiche. `[DA DISCUTERE INSIEME: se vale la pena rieseguire con τ = 0.01 × 3197 come controllo, o dichiararlo come scelta.]`

L'Hessiana esatta è validata in due modi indipendenti prima di costruirci sopra qualsiasi cosa (§5.1 e §5.2).

---

## 5. Validazione della macchina prima dei dati reali

L'ordine di questa sezione è deliberato: nessun numero su HAR è stato interpretato prima che lo strumento fosse collaudato su problemi dove la risposta giusta è nota.

### 5.1 Validazione 1 — riduzione a $K=2$ contro la formula binaria

> **Come funziona questo controllo.** Il codice implementa l'Hessiana nella sua forma generale, valida per un numero qualsiasi di classi. Per il caso di due classi, però, le dispense del corso riportano una formula chiusa, più semplice e ben nota. Se l'implementazione generale è corretta, forzandola a due classi deve riprodurre quella formula fino alla precisione numerica della macchina. È un controllo forte perché non confronta il codice con sé stesso: confronta il codice con un risultato analitico indipendente. Il tipo di verifica che in ingegneria del software si chiama *test di regressione* rispetto a una verità nota.

`notebooks/test_hessian_binary.ipynb`. Problema minimo: 20 dati, 3 caratteristiche, 2 classi, estrattore ridotto all'identità (nessuno strato nascosto, quindi le caratteristiche sono i dati stessi). L'Hessiana generale risulta di dimensione $8\times8$; si estraggono i blocchi $4\times4$ e si confrontano con la formula binaria $\sum_n s_n(1-s_n)\phi_n\phi_n^\top + \tau I$ calcolata separatamente a mano (cella 10).

| confronto | max diff. assoluta | max diff. relativa | esito |
|---|---|---|---|
| $H_{00}$ vs formula binaria | $5.72\times10^{-6}$ | $1.33\times10^{-6}$ | ✓ |
| $H_{11}$ vs formula binaria | $4.77\times10^{-6}$ | $1.11\times10^{-6}$ | ✓ |
| $H_{01} = -H_{00}$ (senza prior) | $7.15\times10^{-7}$ | — | ✓ |

(`test_hessian_binary.ipynb`, celle 14, 16, 18). Le discrepanze sono dell'ordine di $10^{-6}$, cioè al livello dell'errore di arrotondamento dell'aritmetica a precisione singola: coincidenza, non somiglianza.

> **Perché la matrice ha quella struttura a blocchi.** Con due classi la parametrizzazione softmax è ridondante: le due probabilità sommano a uno, quindi basta un numero per descriverle entrambe, ma il codice generale ne mantiene due insiemi di pesi. La ridondanza si manifesta nella struttura: i due blocchi sulla diagonale sono identici fra loro e uguali alla formula binaria, e i blocchi fuori diagonale sono l'opposto dei primi. Ritrovare esattamente questa struttura, e non solo i valori giusti sulla diagonale, è un secondo controllo indipendente sulla correttezza del prodotto di Kronecker implementato. L'implementazione generale a $K$ classi è quindi verificata sul caso in cui esiste una formula chiusa di riferimento nelle dispense.

### 5.2 Validazione 2 — predittiva contro Metropolis–Hastings

`notebooks/test_laplace_vs_mcmc.ipynb`. Problema: 60 dati di addestramento, 20 di verifica, 3 caratteristiche, 3 classi disposte in tre nuvole gaussiane ben separate; la testa ha 12 parametri. La stima puntuale raggiunge accuratezza 0.983 in addestramento e 1.000 in verifica (cella 6). Il campionamento Metropolis–Hastings a passeggiata aleatoria (§3.4) raccoglie 5 000 punti dopo 1 000 di rodaggio, con ampiezza di passo 0.25 e tasso di accettazione 0.299 — dentro l'intervallo considerato ben tarato (cella 10).

> **Che cosa si confronta esattamente.** Per ciascuno dei 20 dati di verifica si calcolano le tre quantità di interesse — incertezza totale, epistemica e aleatoria — in due modi: una volta con l'approssimazione di Laplace, una volta con il campione della distribuzione vera. Poi si guardano due cose distinte, che è importante non confondere. La **correlazione**, che dice se le due misure ordinano i dati nello stesso modo (vale uno se al crescere dell'una cresce sempre l'altra, indipendentemente dalla scala); e la **differenza media in valore assoluto**, che dice quanto i numeri differiscono. Si può avere correlazione altissima e differenze grandi: significa che l'ordinamento è giusto ma la scala è sbagliata. È precisamente quello che accade qui.

Correlazioni di Pearson fra approssimazione di Laplace e campionamento, sui 20 dati di verifica (cella 16):

| quantità | correlazione | diff. assoluta media | diff. assoluta max |
|---|---|---|---|
| entropia totale | 0.939 | 0.324 nat | 0.423 nat |
| epistemica | 0.717 | 0.032 nat | 0.087 nat |
| aleatoria | 0.952 | 0.292 nat | 0.369 nat |
| probabilità medie | — | 0.075 | 0.165 |

Due letture, entrambe importanti e da non confondere.

**L'ordinamento è validato.** Le correlazioni 0.94 e 0.95 su incertezza totale e aleatoria sono alte; quella epistemica è la più debole (0.72), come era da attendersi per due ragioni cumulative: è il termine di grandezza minore, ed è ottenuta come *differenza* fra due quantità stimate per campionamento, quindi eredita il rumore di entrambe. Siccome tutto il progetto usa la componente epistemica come **segnale di graduatoria** — la si correla con l'indicatore di spostamento, e la si standardizza rispetto a riferimenti fissi — e mai come valore assoluto da interpretare, questa è esattamente la proprietà che serve avere validata.

**Il livello assoluto non lo è.** Nella figura di confronto, *tutti* i punti stanno sopra la diagonale di perfetta corrispondenza, per tutte e tre le quantità: la Laplace **sovrastima sistematicamente** l'incertezza rispetto alla posterior campionata su questo problema, di circa 0.29–0.32 nat in media su totale e aleatoria. Non è rumore, è uno scostamento con un segno.
*Figura: `notebooks/test_laplace_vs_mcmc.ipynb`, cella 18 — sezione "Visual Comparison of Uncertainty Estimates" (tre diagrammi a dispersione Laplace vs MCMC con diagonale di riferimento e le correlazioni nei titoli).*

La diagnosi di questo scostamento — una distribuzione a posteriori quasi separabile e non gaussiana — è discussa nelle Limitazioni (§8.2), perché è lì che appartiene concettualmente: non è un bug dell'implementazione, è il carattere locale dell'approssimazione che si manifesta.

### 5.3 Problema giocattolo sintetico in due dimensioni: replica della Fig. 4

> **Perché un problema giocattolo.** Prima di applicare il metodo a dati reali, dove non si sa quale sia la risposta giusta, conviene provarlo su un problema costruito a tavolino, con due sole dimensioni, dove tutto si può disegnare e la risposta giusta è nota per costruzione. In inglese si chiama *toy problem*, «problema giocattolo»: non serve a dimostrare che il metodo funziona nel mondo reale, serve a escludere che sia rotto — e a rendere visibile il meccanismo.

`code_v2/notebooks/02_synthetic_track.ipynb`. Tre classi in due dimensioni (rosso, verde, blu), estrattore a due strati `2 → 32 → 16`, stima puntuale con accuratezza 1.000 in addestramento, matrice di covarianza $51\times51$ con autovalori tutti positivi, nell'intervallo $[0.0290,\, 22.2234]$ (cella 3). Due regimi di spostamento, generati muovendo i centri delle tre nuvole di punti: `mild` («mite»: tutti i centri spostati di poco) e `strong` («forte»: la nuvola blu trascinata in una regione che il modello sorgente chiama con sicurezza «rosso»).

> **A cosa serve il regime mite.** È un **controllo negativo**, cioè un caso in cui il metodo non deve fare *niente*. Il target è già classificato correttamente senza adattamento: se dopo l'adattamento l'accuratezza scendesse, vorrebbe dire che la procedura danneggia situazioni sane. Un esperimento che mostra solo i casi in cui un metodo aiuta, senza mostrare che non nuoce quando non serve, è incompleto.

Risultati (cella 7):

| regime | prima dell'adattamento | dopo adattamento non pesato | dopo adattamento guidato dall'incertezza |
|---|---|---|---|
| mild — accuratezza totale | 1.000 | 1.000 | 1.000 |
| mild — solo classe blu | 1.000 | 1.000 | 1.000 |
| **strong** — accuratezza totale | **0.682** | **0.684** | **0.971** |
| **strong** — solo classe blu | **0.060** | **0.073** | **0.987** |

Questa è la replica, qualitativa e quantitativa, della Fig. 4 del paper: sotto spostamento forte la massimizzazione dell'informazione non pesata non recupera nulla (0.682 → 0.684; sulla sola classe spostata dal 6.0% al 7.3%), mentre la pesatura per incertezza porta la classe blu dal 6.0% al 98.7%. Il controllo negativo mite non peggiora.

> **Perché guardare la sola classe spostata e non solo l'accuratezza totale.** Le tre classi hanno lo stesso numero di punti, quindi due classi classificate perfettamente e una completamente sbagliata danno già un'accuratezza totale attorno al 67%: un numero che sembra mediocre ma non catastrofico, e che nasconde il fatto che una classe è totalmente perduta. Riportare a parte l'accuratezza sulla classe spostata rende visibile il fallimento reale — 6% significa che quasi nessun punto blu è riconosciuto — e il recupero reale.

*Figure: `code_v2/notebooks/02_synthetic_track.ipynb`, cella 7 — sezione "Box 1 -- Fig. 4 replication", due figure 2×3 (una per regime), righe = MAP convenzionale vs guidato dall'incertezza, colonne = Source / Target / SFDA (IM), con superficie di probabilità sfumata in base alla confidenza. Cella 9: mappe affiancate di incertezza epistemica e aleatoria del modello sorgente, con i due punti sonda marcati.*

**Due calibrazioni specifiche del problema giocattolo da dichiarare, non da nascondere** (entrambe documentate nelle celle markdown 2 e 6 del notebook):

1. `TAU_SCALE = 0.01`: la precisione della distribuzione a priori è ridotta a un centesimo del valore prescritto dalla convenzione di §4.5. Al valore nominale la posterior sull'ultimo strato è così concentrata che la patologia da manuale delle reti ReLU si ripresenta quasi ovunque — la *differenza* media fra i logit cresce allontanandosi dai dati tanto quanto la loro deviazione standard, e la confidenza non scende mai in modo apprezzabile con la distanza. Allargare la posterior rende visibile a distanze finite la correzione in stile Kristiadi et al. È una calibrazione del problema giocattolo, non una ricetta generale: il catena di elaborazione su UCI HAR e il controllo MNIST usano la convenzione nominale senza correzioni.
2. `GAMMA = 0.2` invece di 0.5: a $\gamma = 0.5$ il termine di diversità di batch è già da solo abbastanza forte da far ritrovare la nuvola spostata **indipendentemente** dalla pesatura (entrambe le righe convergono alla risposta corretta), il che non dimostra nulla. A $\gamma = 0.2$ il termine di entropia condizionale non pesato domina abbastanza da riprodurre il fallimento del paper. Il recupero mostrato sopra è dunque ottenuto in un regime di iperparametri **scelto perché il fallimento fosse visibile**; non è una previsione su cosa accada a $\gamma = 0.5$ (che è il valore usato nell'ablazione reale, §6.5, dove l'effetto è molto più piccolo).

**Controllo di significato** (cella 11). Qui non si guarda una figura da interpretare: il notebook contiene istruzioni `assert`, cioè verifiche automatiche che interrompono l'esecuzione se la condizione attesa non si verifica — quindi il controllo è un test, non un'impressione.

> **Che cosa si verifica.** La teoria dice che le due componenti dell'incertezza devono rispondere a cose diverse, e si possono costruire due punti in cui si sa quale delle due dovrebbe prevalere. Un punto **lontano da tutte** le nuvole di addestramento: là il modello non ha dati, quindi dovrebbe dominare l'incertezza epistemica. Un punto **dentro** la regione occupata dai dati ma esattamente **sul confine fra due classi**: là il modello ha molti dati e sa bene cosa fare, ma le due classi si sovrappongono davvero, quindi dovrebbe dominare l'incertezza aleatoria. Se le due componenti si comportassero al contrario, la decomposizione sarebbe priva di significato per quanto corretti fossero i conti.

Risultati: il punto lontano, di coordinate $(-4,-4)$, dà epistemica 0.3404 contro aleatoria 0.1490 → **domina l'epistemica**; il punto sul confine verde/blu, di coordinate $(1.0,\, 0.0)$, dà aleatoria 0.7904 contro epistemica 0.0642 → **domina l'aleatoria**. È il comportamento prescritto dalla teoria, verificato su punti scelti geometricamente e non a occhio: il secondo è il punto medio esatto fra due centri sorgente, quindi la sua scelta si giustifica da sé.

**Scansione della severità dello spostamento** (cella 13). Nel report chiamiamo *scansione* (in inglese *sweep*) un esperimento in cui si fa variare un solo parametro per gradi, tenendo tutto il resto fissato, e si osserva come risponde la quantità che interessa. Qui il parametro è l'entità dello spostamento: la nuvola blu viene trascinata **radialmente verso l'esterno**, allontanandola dal centro delle tre classi, in 10 livelli di distanza nota (fino a 3.5).

> **Che ipotesi mette alla prova.** Se l'incertezza epistemica significa davvero «distanza dai dati che ho visto», allora deve crescere in modo ordinato al crescere di una distanza che noi conosciamo esattamente, perché l'abbiamo imposta noi. Non serve che la relazione sia una retta: serve che sia **monotona**, cioè che non torni indietro. La misura appropriata è quindi la correlazione di Spearman, che valuta l'accordo fra due graduatorie anziché fra due valori (§6.4 la spiega in dettaglio).

Risultato: correlazione di Spearman fra entità dello spostamento e incertezza epistemica media sulla classe spostata pari a **0.927**, con probabilità che un accordo così forte nasca dal caso pari a $1.12\times10^{-4}$.

*Figura: `code_v2/notebooks/02_synthetic_track.ipynb`, cella 13 — sezione "Box 3 -- Shift-magnitude sweep", curva epistemica media vs magnitudine di shift con ρ e p nel titolo.*

La scelta della direzione **radiale** è un punto metodologico, non cosmetico, e la motivazione è documentata nel docstring di `src/toy.py::make_classification_sweep` e nella cella markdown 12: muovendo la nuvola di punti verso il centro di un'altra classe la relazione non può essere monotona, perché a un certo punto la nuvola spostata si avvicina di nuovo a densità sorgente e l'epistemica *scende* mentre la magnitudine di shift continua a crescere. Muovendolo radialmente verso l'esterno, il nuvola si allontana monotonamente da ogni nuvola sorgente e la monotonia dell'ipotesi è ben posta. Lo stesso fenomeno si osserva su dati reali con Rotated-MNIST (§5.4): non è un artefatto del problema giocattolo ma una proprietà generale dell'incertezza epistemica, che misura distanza dai dati visti e non "quantità di corruzione".

`[DA DISCUTERE INSIEME: la versione precedente della sweep, con direzione fissa arbitraria e risultato non monotono, non è conservata nella storia git del repository — src/toy.py contiene la versione radiale sin dal primo commit. Se vogliamo raccontarla come risultato osservato e non solo come motivazione di design, serve dirlo in modo attribuibile (per es. "una prima versione, non conservata, mostrava…") oppure ri-eseguirla come controllo esplicito.]`

### 5.4 Cifre scritte a mano contro immagini di abbigliamento, e cifre ruotate

> **Che esperimento è, e perché è il gradino intermedio giusto.** Si addestra un classificatore a riconoscere le cifre scritte a mano del dataset MNIST, e poi gli si mostrano immagini di **capi di abbigliamento** (il dataset Fashion-MNIST: scarpe, magliette, borse), che hanno esattamente lo stesso formato — 28 per 28 pixel in scala di grigi — ma un contenuto completamente estraneo. Un modello che sappia riconoscere la propria ignoranza deve dichiararsi molto più incerto sulle scarpe che sulle cifre. È lo scenario che in inglese si chiama *out-of-distribution detection*, cioè **rilevamento di dati fuori distribuzione**: dati che non provengono dalla popolazione su cui il modello è stato addestrato. Rispetto al problema giocattolo è un caso reale; rispetto a UCI HAR è un caso in cui la risposta giusta è ovvia. Serve a stabilire che il meccanismo funziona *quando lo spostamento è netto*, prima di chiedergli di funzionare quando è sottile.

`notebooks/mnist_check.ipynb`: replica, a una scala eseguibile su un normale processore in meno di un minuto, dell'esperimento di Kristiadi, Hein & Hennig (ICML 2020), riferimento [26] del paper replicato. L'estrattore è qui una piccola **rete convoluzionale** (in inglese *convolutional neural network*: una rete che applica alle immagini filtri locali ripetuti, adatta a dati con struttura spaziale), composta da due blocchi di convoluzione più uno strato completamente connesso, che produce 128 caratteristiche; la testa va da 128 ingressi a 10 uscite, cioè 1 290 parametri trattati in modo bayesiano. Il codice che calcola l'approssimazione di Laplace e la decomposizione dell'incertezza è **identico** a quello del problema giocattolo: cambia soltanto l'estrattore. La distribuzione a priori usa la convenzione nominale di §4.5 senza alcuna riduzione, **a differenza** del problema giocattolo, perché le caratteristiche prodotte dalla rete convoluzionale non mostrano la stessa patologia (cella markdown 4). Covarianza $1290\times1290$ con autovalori tutti positivi; accuratezza sulle cifre di verifica 0.9780 (cella 5).

Confronto fra 2 000 cifre di verifica e 2 000 immagini di abbigliamento (cella 7):

| segnale | MNIST | Fashion-MNIST | rapporto |
|---|---|---|---|
| entropia della stima puntuale (nessun trattamento bayesiano) | 0.1007 | 0.8712 | 8.65× |
| **epistemica** | **0.0502** | **0.5471** | **10.91×** |
| aleatoria | 0.1356 | 0.7878 | 5.81× |
| entropia totale (Laplace) | 0.1858 | 1.3349 | 7.18× |

Le stesse quantità si possono valutare come **rilevatore**, cioè chiedendosi: se dovessi decidere caso per caso se un'immagine è una cifra o un capo di abbigliamento usando solo l'incertezza del modello, quanto ci riuscirei?

> **Come funziona l'area sotto la curva ROC.** Il valore riportato nella tabella seguente è l'**area sotto la curva caratteristica operativa** (in inglese *area under the receiver operating characteristic curve*, abbreviata AUROC). Ha un'interpretazione diretta e facile: è la probabilità che, presi a caso un dato «da rilevare» e un dato «normale», il primo riceva un punteggio di incertezza più alto del secondo. Vale 0.5 se il punteggio è inutile (come lanciare una moneta), 1.0 se separa perfettamente i due gruppi. Il vantaggio di questa misura è che **non dipende dalla scelta di una soglia**: valuta l'ordinamento complessivo, non una particolare regola di decisione. Un valore *sotto* 0.5 non significa «poco informativo», significa che il punteggio è sistematicamente **al contrario** — un fatto che diventerà centrale in §7.

Risultati (cella 10, con le cifre trattate come dati normali e i capi di abbigliamento come dati da rilevare):

| punteggio usato | area sotto la curva ROC |
|---|---|
| entropia della stima puntuale (nessun trattamento bayesiano) | 0.9459 |
| **epistemica** | **0.9871** |
| entropia totale (Laplace) | 0.9791 |

Il punto da sottolineare è **quale** componente porta il segnale. Non si tratta genericamente del fatto che «il trattamento bayesiano aiuta»: è specificamente la componente **epistemica** a dare il rapporto più grande (10.9 volte, contro 8.65 dell'entropia della stima puntuale e 7.18 dell'incertezza totale) e la migliore area sotto la curva (0.987 contro 0.946 e 0.979). L'incertezza totale, che mescola le due componenti, è *peggiore* della sola epistemica su questo compito, perché la parte aleatoria risponde all'ambiguità fra classi e non alla novità del dato — e un capo di abbigliamento non è una cifra ambigua, è una cosa mai vista. La decomposizione non è dunque un ornamento teorico: separa un segnale utile da uno che lo diluisce.

*Figura: `notebooks/mnist_check.ipynb`, cella 8 — sezione "MNIST test vs. Fashion-MNIST test", tre istogrammi sovrapposti (entropia MAP, epistemica BALD, entropia totale) con le due distribuzioni a confronto. Cella 3: campioni di esempio dei due dataset.*

**Cifre ruotate** (cella 12; stesso modello e stessa distribuzione sui pesi, nessun riaddestramento). Qui lo spostamento di dominio è **controllato e continuo**: le stesse cifre di verifica vengono ruotate di un angolo crescente, da 0 a 180 gradi. A differenza del confronto con l'abbigliamento, la severità dello spostamento è ordinabile per costruzione — l'angolo — e questo permette di chiedersi se l'incertezza la segua.

| angolo | 0° | 20° | 40° | 60° | 80° | 100° | 120° | 140° | 160° | 180° |
|---|---|---|---|---|---|---|---|---|---|---|
| accuratezza | 0.978 | 0.905 | 0.585 | 0.256 | 0.160 | **0.116** | 0.118 | 0.210 | 0.287 | 0.306 |
| epistemica | 0.0502 | 0.1105 | 0.2878 | 0.3833 | 0.4342 | **0.4438** | 0.3908 | 0.3464 | 0.2807 | 0.2575 |
| aleatoria | 0.1356 | 0.3105 | 0.6521 | 0.7784 | 0.7669 | 0.7096 | 0.6889 | 0.6783 | 0.6015 | 0.5301 |

L'andamento **non è monotono, e non per un errore**: l'incertezza epistemica cresce ripidamente fino a un massimo attorno ai 100 gradi (0.4438, cioè 8.8 volte il valore a 0 gradi), poi *rientra* verso i 180 gradi (0.2575). La spiegazione, documentata nella cella markdown 11, è che a rotazioni vicine a 180° alcune cifre tornano ad assomigliare a un'altra cifra plausibile (un 6 ruotato somiglia a un 9) anziché a una forma irriconoscibile: il modello torna in una regione dove *ha* visto dati, semplicemente con l'etichetta sbagliata. Lo specchio si vede nell'accuratezza, che ha il minimo a 100–120° (0.116) e risale parzialmente a 0.306 a 180°. È esattamente lo stesso fenomeno che motiva la direzione radiale nella sweep del problema giocattolo (§5.3): l'epistemica misura distanza dal supporto dei dati, non gravità della corruzione.

*Figura: `notebooks/mnist_check.ipynb`, cella 13 — sezione "Part 2 -- Rotated MNIST", due pannelli (incertezze vs angolo; accuratezza vs angolo) più una striscia di cifre di esempio lungo la sweep.*

La tabella è verificabile direttamente nell'output salvato della cella 12 del notebook come sta su disco (esecuzione sequenziale completa, `execution_count` 1–8), e i valori coincidono con quelli citati in `TODO.md` §3b.

---

## 6. Risultati su UCI HAR

### 6.1 Addestramento del modello sorgente (stima puntuale)

> **Che cos'è l'addestramento a stima puntuale.** È l'addestramento ordinario, non bayesiano: si cerca l'unica combinazione di pesi che massimizza la probabilità dei dati osservati tenendo conto anche della penalizzazione sui pesi grandi. Il risultato è la stima puntuale su cui, in un secondo momento e senza rifare nulla, si innesta il trattamento bayesiano dell'ultimo strato. Questa separazione è uno dei vantaggi rivendicati dal paper: il modello sorgente si addestra come si è sempre fatto.

`notebooks/har_source_training.ipynb`. Dentro l'insieme sorgente si separano ulteriormente 16 soggetti per l'addestramento e 4 per la verifica interna (i soggetti `1, 2, 24, 26`), sempre dividendo per soggetto e non per finestra: 2 510 finestre di addestramento, 687 di verifica (cella 7).

> **Come funzionano gli ingredienti dell'addestramento.** L'**ottimizzatore** AdamW modifica i pesi a piccoli passi nella direzione che riduce l'errore, adattando automaticamente l'ampiezza del passo per ciascun peso; il **passo di apprendimento** (in inglese *learning rate*), qui $10^{-3}$, ne fissa la scala complessiva. Il **decadimento dei pesi** (0.01) è la penalizzazione sui pesi grandi già discussa in §4.5. Un'**epoca** è un passaggio completo su tutti i dati di addestramento. L'**arresto anticipato** (in inglese *early stopping*) serve a non memorizzare i dati: dopo ogni epoca si misura l'errore sui 4 soggetti tenuti da parte e, se non migliora per 20 epoche consecutive — la cosiddetta *pazienza* — l'addestramento si interrompe e si torna ai pesi dell'epoca migliore. Qui l'arresto è avvenuto all'epoca 111, con l'epoca migliore risultata la 91 (cella 15).

> **Perché si controlla il bilanciamento delle classi.** Se una classe fosse molto più frequente delle altre, il modello potrebbe ottenere buona accuratezza ignorandola quasi del tutto. Il controllo verifica il rapporto fra la classe più frequente e la meno frequente **dentro l'insieme di soggetti effettivamente scelto**, non solo sul dataset globale — perché la scelta dei soggetti potrebbe da sola aver creato uno sbilanciamento. Qui i conteggi sono 910, 839 e 761, con un rapporto di 1.20: abbastanza bilanciato perché nessuna correzione sia necessaria (cella 11).

Sulla verifica interna: **accuratezza 0.975 e richiamo macro-medio 0.974** (cella 17).

> **Che cos'è il richiamo macro-medio.** L'accuratezza è la frazione di risposte giuste sul totale. Il **richiamo** di una classe è invece la frazione di dati *di quella classe* che il modello riconosce correttamente; il **richiamo macro-medio** (in inglese *macro-recall*) è la media semplice dei richiami delle tre classi, che dà a ciascuna classe lo stesso peso indipendentemente da quanto è frequente. È riportato accanto all'accuratezza perché è la misura che si accorge del disastro che l'accuratezza può nascondere: un modello che ignorasse completamente una classe avrebbe ancora buona accuratezza ma richiamo macro-medio molto più basso. Qui i due valori praticamente coincidono, il che è la conferma cercata.

Per soggetto target (cella 19):

| soggetto | 7 | 8 | 11 | 15 | 19 | 20 | 21 | 22 | 27 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|
| accuratezza | 0.935 | 0.913 | 0.987 | 0.993 | 0.779 | **0.653** | 0.979 | 0.895 | **1.000** | 0.995 |
| macro-recall | 0.936 | 0.924 | 0.989 | 0.994 | 0.804 | 0.643 | 0.980 | 0.902 | 1.000 | 0.995 |

Accuratezza media 0.913 ± 0.108, **intervallo 0.653–1.000**. Il range è il dato importante, più della media: non c'è saturazione (esiste almeno un soggetto duro, il 20) e non c'è collasso al livello del caso (il minimo 0.653 è ben sopra 1/3). Ma va detto subito che **6 dei 10 target sono già ≥ 0.94 senza alcun adattamento**: la popolazione target è dominata da soggetti su cui non c'è nulla da guadagnare. Questa asimmetria è la ragione strutturale per cui la media macro dell'ablazione (§6.5) è fuorviante.

*Figura: `notebooks/har_source_training.ipynb`, cella 17 — sezione "7. Training Curves", tre pannelli (cross-entropy, accuratezza, macro-recall) train vs val con l'epoca migliore evidenziata.*

### 6.2 Convergenza Monte Carlo: un criterio tautologico trovato, corretto, e non soddisfatto

> **Perché serve un criterio per $M$.** La predizione bayesiana è una media su $M$ estrazioni casuali di pesi (§3.3). Con $M$ piccolo la media è rumorosa: ripetendo il calcolo si otterrebbero numeri diversi, e non si saprebbe se una differenza fra due soggetti sia reale o solo un artefatto del caso. Con $M$ grande il rumore si riduce ma il costo cresce in proporzione. «Convergenza» significa qui aver trovato un $M$ oltre il quale aumentarlo non cambia più il risultato in modo apprezzabile: da quel punto in poi, ciò che si misura è la quantità vera e non il rumore di campionamento. Scegliere $M$ a occhio sarebbe accettabile solo se poi non si interpretassero differenze piccole — cosa che invece qui si fa.

`notebooks/har_mc_convergence.ipynb`. Il criterio inizialmente naturale — «confronta ogni $M$ con il valore al massimo $M$ testato» — è **tautologico**: il valore più grande della lista viene confrontato con sé stesso e passa per costruzione. La versione corretta (documentata nella cella markdown 14) usa un riferimento **indipendente**, $M = 5000$, che non appartiene alla lista dei valori esaminati, e richiede tre condizioni contemporaneamente: scostamento relativo dal riferimento inferiore all'1%, scostamento assoluto inferiore a 0.0005, e stabilità su una finestra di tre valori consecutivi.

> **A che serve ciascuna delle tre condizioni.** Lo scostamento **relativo** è la condizione naturale, ma da sola inganna quando la quantità misurata è minuscola: l'1% di un numero molto piccolo è una soglia facilissima da rispettare per puro caso. Lo scostamento **assoluto** mette un pavimento e impedisce questo. La **finestra di tre valori** serve contro le coincidenze: con quantità rumorose può succedere che un singolo valore di $M$ finisca vicino al riferimento per fortuna, mentre è improbabile che ci finiscano tre valori consecutivi. Il seme del generatore casuale è tenuto fisso (123) per tutti i valori di $M$, così che le differenze osservate vengano dal numero di estrazioni e non dal cambiare l'estrazione.

Il modello e la posterior sono ricostruiti deterministicamente da §4 e verificati: $\tau_{\text{prior}}$ identico a 25.10 (differenza 0.00) e autovalore minimo della covarianza coerente con §4 (celle 10 e 11).

Risultato (cella 19):

| $M$ | 10 | 25 | 50 | 100 | 200 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|---|---|---|
| dev. rel. su verifica sorgente | 23.93% | 18.20% | 14.31% | 1.18% | 1.62% | 0.23% | 1.39% | 0.12% |
| dev. rel. su target 21 | 0.76% | 7.08% | 2.56% | 2.55% | 1.38% | 1.03% | 0.30% | 1.03% |
| criterio puntuale | no | no | no | no | no | no | no | no |

**Nessun $M$ soddisfa il criterio rigoroso.** Le deviazioni relative oscillano intorno all'1% senza scendere sotto in modo stabile su tre valori consecutivi: il rumore di campionamento domina sull'andamento di convergenza a tutti i valori di $M$ esaminati, perché il segnale epistemico su questo dataset è piccolo in assoluto (~$5.8\times10^{-3}$ nat) e stabilizzare una quantità di quell'ordine richiede molti campioni. Il notebook prende la strada onesta: stampa «NO M in M_VALUES satisfied windowed convergence criterion» e adotta $M_{\text{FIXED}} = 5000$ = il riferimento stesso, cioè il valore più alto disponibile come stima più affidabile, dichiarando che il criterio *non* è stato soddisfatto.

Questo è riportato qui come esempio di rigore, non come risultato positivo travestito. Un criterio che *non* si supera e viene dichiarato tale è più informativo di uno tarato a posteriori per essere superato. La conseguenza pratica è che tutte le quantità epistemiche di §6.3 sono calcolate con $M = 5000$ e che l'incertezza di campionamento che resta su di esse è dell'ordine dell'1% relativo — non trascurabile in assoluto, ma piccola rispetto alle differenze fra soggetti (che sono di ordini di grandezza, si veda §6.3).

*Figura: `notebooks/har_mc_convergence.ipynb`, cella 18 — sezione "4. Convergence Analysis and M Selection", griglia 2×3 (verifica sorgente e target 21 × totale/epistemica/aleatoria) in funzione di $M$ in scala logaritmica, con la linea del riferimento $M = 5000$.*

### 6.3 Scomposizione dell'incertezza

`notebooks/har_uncertainty_decomposition.ipynb`, sull'artefatto di §6.2 ($M = 5000$, temperatura 1.0). Cella 7:

| soggetto | n | accuratezza | epistemica | aleatoria | totale |
|---|---|---|---|---|---|
| verifica sorgente | 687 | 0.975 | 0.0058 | 0.0591 | 0.0649 |
| 7 | 155 | 0.935 | 0.0167 | 0.0760 | 0.0927 |
| 8 | 127 | 0.913 | 0.0388 | 0.1413 | 0.1801 |
| 11 | 159 | 0.987 | 0.0059 | 0.0394 | 0.0454 |
| 15 | 144 | 0.993 | 0.0027 | 0.0242 | 0.0269 |
| 19 | 131 | 0.779 | 0.0447 | 0.1189 | 0.1636 |
| **20** | 147 | **0.653** | **0.0515** | 0.1782 | 0.2298 |
| 21 | 144 | 0.979 | 0.0046 | 0.0358 | 0.0405 |
| 22 | 124 | 0.895 | 0.0228 | 0.0645 | 0.0872 |
| 27 | 152 | 1.000 | 0.0002 | 0.0018 | 0.0020 |
| 30 | 192 | 0.995 | 0.0025 | 0.0293 | 0.0318 |

Due osservazioni immediate. Primo: **l'aleatoria è più grande dell'epistemica dappertutto**, anche sul soggetto peggiore. Con 3 classi il tetto di entropia è $\log 3 = 1.10$ nat e le classi di locomozione hanno sovrapposizione reale già appartenente al dominio noto; non c'è molto spazio per nessuno dei due termini. Secondo: è comunque **l'epistemica** il segnale che traccia la difficoltà.

**Controllo di significato A — soggetto più difficile → epistemico-dominante** (cella 10, con `assert`). Il soggetto con l'accuratezza più bassa (20, 0.653) è **anche** quello con l'epistemica più alta di tutta la coorte (0.0515), pari a **8.9×** il riferimento del verifica interna sorgente (0.0058). Il controllo è formulato come *ranking*, non come dominanza assoluta dell'epistemica sull'aleatoria (che, per il punto precedente, non si verifica mai qui) — ed è la formulazione corretta della proprietà che si vuole verificare.

*Figura: `notebooks/har_uncertainty_decomposition.ipynb`, cella 11 — sezione "3. Semantic check -- hardest target subject", due pannelli: diagramma a dispersione epistemica vs accuratezza con etichette dei soggetti, e barre orizzontali di epistemica media ordinate (verifica sorgente incluso).*

**Controllo di significato B — SITTING vs STANDING → aleatorio-dominante** (celle 13–15). Il modello a 3 classi non conosce le posture statiche, quindi questo controllo richiede un modello dedicato: stessa architettura ma `561 → 128 → 64 → 6`, stesso split per soggetto (verificato identico con `assert`), scaler rifittato, $\tau_{\text{prior}} = 54.42$, accuratezza su val sorgente 0.970. SITTING e STANDING sono l'esempio da manuale di incertezza *aleatoria* in HAR da accelerometro: due posture statiche con statistiche del segnale quasi identiche.

| gruppo | epistemica | aleatoria | rapporto ale/epi |
|---|---|---|---|
| SITTING + STANDING | 0.0126 | 0.1082 | 8.6× |
| classi ben separabili (WALKING, DOWNSTAIRS, LAYING) | 0.0050 | 0.0190 | 3.8× |
| **elevazione rispetto alle ben separabili** | **2.5×** | **5.7×** | — |

La confusione fra le due posture è dunque guidata dall'aleatoria (elevata 5.7×) e non dall'epistemica (2.5×): è ambiguità intrinseca delle caratteristiche, non mancanza di dati. La matrice di confusione conferma la localizzazione dell'errore (SITTING → STANDING 7 casi, STANDING → SITTING 8, SITTING → LAYING 13; le altre classi quasi diagonali, cella 14).

*Figura: `notebooks/har_uncertainty_decomposition.ipynb`, cella 16 — sezione "4. Semantic check -- SITTING vs STANDING", barre epistemica/aleatoria per ognuna delle 6 classi, con SITTING e STANDING evidenziate.*

**Riportare l'incertezza su una scala assoluta** (celle 18–19). I valori grezzi (da 0.0002 a 0.05 nat) non dicono nulla presi da soli: 0.05 è molto o poco? Serve un termine di paragone, e la scelta di quale usare è una decisione metodologica con conseguenze.

> **Perché non si normalizza rispetto al gruppo di dati corrente.** La tentazione naturale sarebbe dividere per la media dei dati che si stanno esaminando in quel momento — in inglese si direbbe normalizzare *per batch*, dove un *batch* è il gruppo di dati processati insieme. Sarebbe attivamente ingannevole: un gruppo composto tutto da soggetti difficili apparirebbe «normale» rispetto a sé stesso, e il segnale che si vuole misurare — il fatto che quei soggetti *sono* difficili — verrebbe cancellato proprio dalla normalizzazione. Il piano di lavoro lo vieta esplicitamente (`TODO.md` §1b: «never per-batch», mai rispetto al gruppo corrente).

La funzione `normalize_epistemic` usa invece due riferimenti **fissi**, calcolati una volta sola e riutilizzati identici per ogni soggetto: la divisione per $\log K = 1.0986$, cioè per l'entropia massima possibile con tre classi (un riferimento che non dipende da alcun dato), e un punteggio standardizzato robusto rispetto alla distribuzione dell'incertezza epistemica sulla **verifica interna sorgente**.

> **Come funziona il punteggio standardizzato robusto.** Il punteggio standardizzato usuale sottrae la media e divide per la deviazione standard, rispondendo alla domanda: «quante deviazioni standard sopra il normale sta questo valore?». Nella versione **robusta** si usano al posto di media e deviazione standard due quantità che non si lasciano trascinare dai valori estremi: la **mediana** (il valore centrale, che divide il campione in due metà) e lo **scarto interquartile** (in inglese *interquartile range*: la distanza fra il valore che supera un quarto dei dati e quello che ne supera tre quarti). Qui la mediana di riferimento è 0.000453 e lo scarto interquartile 0.002205, entrambi calcolati sulla sola verifica sorgente. Il risultato è leggibile su una scala unica: un punteggio vicino a zero significa «indistinguibile dai dati che il modello conosce», un punteggio di 20 significa «venti volte lo scarto tipico oltre il normale».

| soggetto | 27 | 30 | 15 | 21 | verifica sorgente | 11 | 7 | 22 | 8 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| z-score | −0.12 | 0.94 | 1.02 | 1.89 | 2.43 | 2.49 | 7.36 | 10.12 | 17.40 | 20.06 | **23.17** |

Su una scala fissata una volta sola dai dati sorgente, i soggetti facili sono indistinguibili dal rumore del verifica interna sorgente ($z \lesssim 2.5$) e quelli difficili stanno a $z > 17$: un rapporto di ordini di grandezza, non un effetto marginale.

*Figura: `notebooks/har_uncertainty_decomposition.ipynb`, cella 19 — sezione "5. Absolute normalization", barre orizzontali dei z-score ordinati con la linea dello zero.*

**Effetto di $\tau$** (celle 21–22). Qui $\tau$ è la `temperature` della predittiva (l'iperparametro fissato a 0.4 in `TODO.md` §1), non la precisione della prior. Sweep su verifica sorgente e sul soggetto più difficile:

| $\tau$ | 0.1 | 0.2 | **0.4** | 0.7 | 1.0 | 1.5 | 2.0 | 4.0 |
|---|---|---|---|---|---|---|---|---|
| frazione epistemica, verifica sorgente | 0.757 | 0.555 | **0.303** | 0.147 | 0.090 | 0.057 | 0.044 | 0.023 |
| frazione epistemica, soggetto 20 | 0.873 | 0.753 | **0.552** | 0.348 | 0.230 | 0.130 | 0.084 | 0.029 |

La frazione epistemica cala **monotonamente** da $\tau = 0.1$ a $\tau = 4$ per entrambi. Il meccanismo (cella markdown 23): dividere i logit per $\tau < 1$ rende ogni singolo campione della posterior più vicino a one-hot (entropia per campione più bassa → aleatoria giù) e amplifica il disaccordo *fra* campioni (epistemica su); oltre $\tau = 1$ entrambi gli effetti si invertono. Il valore fissato $\tau = 0.4$ tiene l'epistemica intorno a metà del segnale totale per il soggetto difficile (0.552), contro solo il 23% al default non toccato $\tau = 1$, senza l'instabilità del regime $\tau < 0.2$ dove la decomposizione diventa quasi tutta epistemica e smette di discriminare fra soggetti.

*Figura: `notebooks/har_uncertainty_decomposition.ipynb`, cella 22 — sezione "6. Effect of τ", due pannelli: frazione epistemica vs τ (scala log) con la linea a τ = 0.4, e magnitudini grezze su verifica sorgente.*

### 6.4 Calibrazione: il risultato chiave

> **Che cos'è la calibrazione, e perché è diversa dall'accuratezza.** Un modello è **accurato** se indovina spesso; è **calibrato** se le probabilità che dichiara corrispondono alle frequenze reali. Sono cose indipendenti. Un modello che dice sempre «sono sicuro al 70%» e ha ragione nel 70% dei casi è perfettamente calibrato pur non essendo bravissimo; un modello che dice sempre «sono sicuro al 99%» e ha ragione nell'80% dei casi è accurato ma **sovraconfidente**, e il suo eccesso di sicurezza è pericoloso proprio perché invisibile guardando la sola accuratezza. Questa sezione misura se il trattamento bayesiano produce probabilità più oneste, non solo decisioni migliori — che è la promessa specifica che il paper fa.

`notebooks/har_calibration.ipynb`.

> **Come funziona il protocollo a più semi.** Il piano di lavoro impone almeno 5 semi per qualunque correlazione o graduatoria riportata (`TODO.md` §1). Ma bisogna decidere *che cosa* far variare fra un seme e l'altro. Qui la divisione fra soggetti sorgente e target, e quella fra addestramento e verifica interna, restano **fisse**: cambiare chi sta in quale gruppo cambierebbe il problema, non la sua soluzione, e mediare su problemi diversi non risponderebbe a nessuna domanda. Ciò che varia sui 5 semi (da 0 a 4) è l'inizializzazione casuale dei pesi e quindi il percorso di addestramento: cinque modelli diversi, addestrati sugli stessi dati, che danno cinque stime di calibrazione e di incertezza. La dispersione fra queste cinque stime è la barra d'errore riportata dappertutto nel seguito, e misura esattamente una cosa: quanto un risultato dipende dalla fortuna dell'inizializzazione.

Impostazioni: $M = 1000$ estrazioni di pesi, errore di calibrazione calcolato su 15 intervalli, diagrammi di affidabilità disegnati con 8 intervalli e con l'area del simbolo proporzionale al numero di dati che cadono nell'intervallo, così che un punto basato su pochi dati si legga come rumore e non come segnale.

#### Che cosa misurano le quattro metriche usate

> **Diagramma di affidabilità** (in inglese *reliability diagram*). È il grafico che rende visibile la calibrazione. Si raggruppano le predizioni per livello di confidenza dichiarata — per esempio tutte quelle fra il 70% e l'80% — e per ciascun gruppo si confronta la confidenza media dichiarata con la frazione di risposte effettivamente corrette. Se il modello è calibrato i punti cadono sulla diagonale; se stanno **sotto** la diagonale il modello promette più di quanto mantiene, cioè è sovraconfidente.

> **Errore di calibrazione atteso** (in inglese *expected calibration error*, sigla ECE). È il riassunto numerico del diagramma: la media, pesata per quanti dati contiene ciascun intervallo, della distanza verticale fra i punti e la diagonale. Vale zero per un modello perfettamente calibrato. Ha un limite noto e va ricordato: dipende dal numero di intervalli scelto, e non distingue un errore in eccesso da uno in difetto.

> **Verosimiglianza logaritmica negativa** (in inglese *negative log-likelihood*, sigla NLL). È la media del logaritmo negativo della probabilità che il modello assegnava alla risposta **giusta**. Premia le probabilità alte sulla classe corretta e punisce in modo severissimo la sicurezza sbagliata: se il modello dà probabilità quasi nulla alla classe che poi risulta vera, il logaritmo tende a meno infinito e il valore esplode. È per questo che la variante collassato dello studio di §6.5 raggiunge valori intorno a 13, mentre le varianti sani stanno sotto 4.

> **Punteggio di Brier** (in inglese *Brier score*). È l'errore quadratico medio fra il vettore di probabilità dichiarato e la risposta vera codificata come vettore con un uno e il resto zeri. Rispetto alla verosimiglianza logaritmica è meno sensibile ai casi estremi, perché la penalità cresce come un quadrato e non come un logaritmo: le due misure insieme dicono se un risultato è guidato da poche predizioni catastrofiche (la prima peggiora molto più della seconda) o da un errore diffuso (peggiorano insieme).

Risultati: errore di calibrazione, verosimiglianza logaritmica negativa e punteggio di Brier, media ± deviazione standard sui 5 semi (celle 12–13):

| soggetto | accuratezza | errore calibr. (puntuale) | errore calibr. (Laplace) | verosim. log. neg. (puntuale) | verosim. log. neg. (Laplace) | Brier (puntuale) | Brier (Laplace) |
|---|---|---|---|---|---|---|---|
| verifica sorgente | 0.9758 ± 0.0027 | 0.0104 ± 0.0030 | 0.0113 ± 0.0024 | 0.0504 | 0.0516 | 0.0305 | 0.0305 |
| 7 | 0.9458 ± 0.0139 | 0.0441 | **0.0392** | 0.1777 | **0.1626** | 0.0855 | 0.0834 |
| 8 | 0.9228 ± 0.0261 | 0.0626 | **0.0547** | 0.2528 | **0.2223** | 0.1198 | 0.1165 |
| 11 | 0.9799 ± 0.0047 | 0.0162 | **0.0149** | 0.0621 | **0.0568** | 0.0310 | 0.0303 |
| 15 | 0.9931 ± 0.0000 | 0.0091 | 0.0114 | 0.0153 | 0.0168 | 0.0095 | 0.0090 |
| **19** | 0.7679 ± 0.0142 | 0.2006 | **0.1729** | 1.1915 ± 0.1223 | **0.7200 ± 0.0636** | 0.4009 | **0.3539** |
| **20** | 0.6803 ± 0.0122 | 0.2676 | **0.2381** | 1.5475 ± 0.0833 | **1.0268 ± 0.0529** | 0.5537 | **0.5097** |
| 21 | 0.9833 ± 0.0034 | 0.0163 | **0.0150** | 0.0301 | **0.0288** | 0.0203 | 0.0188 |
| 22 | 0.8887 ± 0.0119 | 0.0901 | **0.0814** | 0.4164 | **0.3106** | 0.1802 | 0.1658 |
| 27 | 1.0000 ± 0.0000 | 0.0002 | 0.0003 | 0.0002 | 0.0003 | 0.0000 | 0.0000 |
| 30 | 0.9948 ± 0.0000 | 0.0067 | 0.0088 | 0.0164 | 0.0169 | 0.0094 | 0.0092 |

La Laplace migliora l'ECE media su **7 soggetti su 11** e la NLL media su **7 su 11** (cella 13), e sono i soggetti `7, 8, 11, 19, 20, 21, 22`. I guadagni grandi sono concentrati sui difficili: soggetto 20 NLL 1.55 → 1.03, soggetto 19 NLL 1.19 → 0.72. Due dei sette (11 e 21) sono però soggetti facili con miglioramento piccolo, quindi la formulazione precisa è «i guadagni sono concentrati sui difficili», non «i miglioramenti avvengono solo sui difficili».

Sui quattro rimanenti (verifica sorgente, 15, 27, 30) la Laplace **peggiora leggermente** un ECE già minuscolo (per es. verifica sorgente 0.0104 → 0.0113): con un modello già ben calibrato, la dispersione aggiuntiva della posterior sovracorregge. È un risultato onesto e coerente col meccanismo: la Laplace guadagna dove la stima puntuale è troppo confidente, e non ha niente da correggere dove già non lo è.

*Figura: `notebooks/har_calibration.ipynb`, cella 10 — sezione "4. Reliability diagrams, MAP vs. Laplace", griglia 3×4 con un pannello per verifica sorgente e per ognuno dei 10 target (predizioni aggregate sui 5 semi, MAP in rosso e Laplace in blu, area del simbolo = conteggio dell'intervallo, indicatore di spostamento nel titolo).*

**Il risultato chiave: correlazione di Spearman pari a 0.84 ± 0.03** (cella 21).

> **Come funziona la correlazione di Spearman, e perché è quella giusta qui.** La correlazione di Pearson misura quanto due grandezze stanno su una retta. Quella di **Spearman** misura invece se le due grandezze **ordinano** gli oggetti nello stesso modo: si sostituisce a ciascun valore il suo posto in graduatoria e si calcola la correlazione fra le due graduatorie. Vale 1 se l'ordine è identico, 0 se non c'è relazione, −1 se è esattamente rovesciato. È la misura appropriata in questo caso per due ragioni. Primo: non c'è motivo di aspettarsi che l'incertezza cresca *linearmente* con la distanza fra centroidi — l'ipotesi da verificare è solo che cresca. Secondo: l'indicatore di spostamento è grezzo, e una misura basata sugli ordini è meno sensibile a un singolo valore fuori scala di quanto lo sarebbe una misura basata sui valori.

> **Come si legge il valore p.** Accanto alla correlazione si riporta il **valore p**: la probabilità di osservare un accordo così forte, o più forte, se in realtà le due grandezze fossero indipendenti. Un valore p di 0.0016 significa che un accordo del genere nascerebbe dal caso in circa due casi su mille. Con soli 10 soggetti questo controllo non è un dettaglio: una correlazione alta su pochi punti può ben essere una coincidenza, e il valore p è ciò che permette di escluderlo.

> **Perché si media il coefficiente e non i valori.** Per ciascun seme si calcola *separatamente* la correlazione fra i due ordinamenti dei 10 soggetti target, e solo dopo si fa la media delle cinque correlazioni ottenute. L'alternativa — mediare prima le incertezze dei cinque modelli e poi calcolare un'unica correlazione — darebbe un numero solo, senza barra d'errore, e nasconderebbe proprio l'informazione cercata: se il risultato regga su ogni singolo modello o dipenda dalla fortuna dell'inizializzazione. È letteralmente ciò che il piano di lavoro chiede (`TODO.md` §1).

Risultati per seme:

| seme | 0 | 1 | 2 | 3 | 4 | media ± dev. std |
|---|---|---|---|---|---|---|
| ρ di Spearman | 0.855 | 0.867 | 0.855 | 0.855 | 0.782 | **0.842 ± 0.031** |
| p | 0.0016 | 0.0012 | 0.0016 | 0.0016 | 0.0075 | tutti < 0.01 |

Il coefficiente è stabile (deviazione standard 0.031 su cinque inizializzazioni indipendenti) e significativo su ogni seme preso singolarmente. Per confronto, la stessa correlazione con l'ECE della Laplace, calcolata sulle medie fra semi, vale ρ = 0.806 (p = 0.0049) (cella 24) — anche la calibrazione degrada con lo shift, ma l'epistemica è il segnale più netto dei due.

Un disaccordo merita di essere segnalato, non appianato: il soggetto **20 ha un indicatore di spostamento *minore* del 19** (18.43 vs 29.21) ma **epistemica maggiore** (0.0515 vs 0.0447) e accuratezza minore (0.653 vs 0.779). Il proxy a centroide su 561 dimensioni non è la verità di riferimento; qui l'epistemica traccia la difficoltà *reale* meglio dell'indicatore. Detto in modo rigoroso: ρ = 0.84 misura l'accordo con un proxy grezzo, e l'unico caso in cui i due si discostano è un caso in cui l'indicatore sbaglia. Questo rafforza il risultato, ma va enunciato come osservazione su un singolo soggetto, non come dimostrazione.

*Figure: `notebooks/har_calibration.ipynb`, cella 22 — sezione "8. Spearman(shift proxy, epistemic uncertainty)", diagramma a dispersione con barre d'errore sui 5 semi ed etichette dei soggetti. Cella 24 — sezione "9. Summary plots", due pannelli affiancati: ECE vs shift ed epistemica vs shift, stesso asse x.*

**La sfumatura sul rilevamento degli errori** (celle 16–17).

> **Che cosa si misura qui, e in che cosa differisce da §5.4.** Là si chiedeva all'incertezza di distinguere dati appartenenti a popolazioni diverse; qui le si chiede una cosa più difficile: distinguere, **dentro** lo stesso insieme di dati, le predizioni giuste da quelle sbagliate. Si etichetta ogni predizione con «corretta» o «errata» e si usa l'incertezza come punteggio per separare le due, misurando di nuovo l'area sotto la curva. Un valore alto significa che il modello, quando sbaglia, tende effettivamente a essere anche incerto — la proprietà che rende possibile astenersi al momento giusto. Nota di lettura: per il soggetto 27 il valore è indefinito e riportato come tale, perché il modello non commette **nessun** errore su quel soggetto e la misura richiede la presenza di entrambi i casi.

Il quadro è più articolato che nelle sezioni precedenti, e va riportato per intero:

| soggetto | area curva, entropia puntuale | area curva, incertezza totale | area curva, epistemica |
|---|---|---|---|
| verifica sorgente | **0.9880** | 0.9869 | 0.9773 |
| 7 | **0.8933** | 0.8912 | 0.8686 |
| 8 | **0.8729** | 0.8632 | 0.8508 |
| 11 | 0.9652 | 0.9646 | 0.9641 |
| 15 | 0.9944 | 0.9972 | **1.0000** |
| **19** | 0.8020 | 0.8573 | **0.8930** |
| **20** | 0.7331 | 0.7701 | **0.8109** |
| 21 | 0.9948 | 0.9953 | **0.9972** |
| 22 | 0.9180 | **0.9299** | 0.9259 |
| 27 | nan (0 errori) | nan | nan |
| 30 | **0.9895** | 0.9895 | 0.9895 |

L'epistemica **non** è uniformemente il miglior rilevatore di errore. Sui soggetti 7 e 8 — moderatamente spostati secondo l'indicatore ma ancora accurati al 92–95% — l'entropia MAP e la totale la superano leggermente, perché lì la maggior parte degli errori nasce da sovrapposizione reale fra classi, cioè è aleatoria. Sui due soggetti con l'accuratezza **più bassa in assoluto** — 20 (68%) e 19 (77%), i davvero difficili per esito — l'epistemica è invece la **migliore** delle tre (20: 0.811 > 0.770 > 0.733; 19: 0.893 > 0.857 > 0.802).

La formulazione corretta è quindi: **l'epistemica è prima un rilevatore di shift, poi di errore.** Traccia in modo affidabile la difficoltà del dominio (§6.3, e ρ = 0.84 qui sopra); se sia anche il miglior rilevatore di errore su un dato soggetto dipende da quale tipo di incertezza domini gli errori là. È una distinzione che l'entropia totale, mescolando i due termini, non permette di fare.

**Accuratezza in funzione della copertura** (cella 19).

> **Come funziona questa curva, e perché è la misura più vicina all'uso reale.** Le metriche precedenti presuppongono che il modello debba rispondere sempre. Ma un sistema che sa di essere incerto potrebbe **astenersi**: rispondere solo quando si fida, e in caso contrario chiedere conferma o passare la mano. La **copertura** è la frazione di casi su cui il sistema decide di rispondere. La curva si costruisce ordinando tutti i dati dal più certo al più incerto, e calcolando l'accuratezza sul primo 10%, sul primo 20%, e così via fino al 100%. Se l'incertezza è informativa, l'accuratezza deve **salire** al ridursi della copertura: rinunciando ai casi dubbi si sbaglia meno su quelli che restano. Se la curva fosse piatta, l'incertezza non conterrebbe informazione utile a decidere quando tacere. Questa impostazione si chiama in inglese *selective prediction*, «predizione selettiva», ed è il modo naturale di tradurre una misura di incertezza in un comportamento operativo.

Ordinando i dati per incertezza totale crescente e calcolando l'accuratezza sulla frazione più affidabile, la curva sale al ridursi della copertura anche sul soggetto con lo spostamento maggiore: il soggetto 19 parte da 0.768 a copertura piena (valore dalla tabella a più semi sopra) e la curva tende verso ≈1.0 a copertura bassa. **Il valore ≈1.0 è una lettura della curva, non un numero stampato** dal notebook (`TODO.md` §8 riporta la stessa lettura). Operativamente è il risultato più direttamente rilevante per uno scenario assistivo: se il sistema potesse astenersi quando è incerto, l'accuratezza sulle risposte che decide di dare tornerebbe vicina a quella sorgente.

*Figura: `notebooks/har_calibration.ipynb`, cella 19 — sezione "7. Accuracy-vs-coverage curve", quattro curve (verifica sorgente e i soggetti a shift basso/medio/alto: 27, 11, 19), reiezione per entropia totale della Laplace.*

### 6.5 Studio comparativo per rimozione (in inglese *ablation study*)

> **Che cos'è un'ablazione, e perché si chiama così.** Il termine inglese *ablation* — dal latino *ablatio*, «asportazione» — è preso in prestito dalla fisiologia, dove indica la rimozione chirurgica di una parte per capire a cosa serviva. In apprendimento automatico indica un esperimento in cui si smonta un metodo pezzo per pezzo: si eseguono più versioni della stessa procedura, **identiche in tutto** tranne per la presenza o l'assenza di un singolo componente, e si confrontano i risultati. Se togliendo un pezzo la prestazione crolla, quel pezzo serviva; se non cambia nulla, il pezzo era decorativo. Nel report chiamiamo **variante** (in inglese *arm*, letteralmente «braccio», per analogia con i bracci di una sperimentazione clinica) ciascuna delle versioni messe a confronto.

> **Perché è il cuore dell'esperimento.** Il paper attribuisce il proprio vantaggio a una cosa precisa: pesare i dati target con l'incertezza stimata da un modello **bayesiano**. Mostrare che il metodo completo funziona non basta a dimostrarlo, perché il merito potrebbe stare altrove — nel pesare i dati *in qualunque modo*, o nel solo termine di diversità. Le sele varianti sono costruite per isolare esattamente il contributo di ciascun ingrediente, e la variante (d) — che pesa con l'incertezza del modello **non** bayesiano — è il confronto decisivo: se (d) andasse quanto (e), il trattamento bayesiano non aggiungerebbe niente.

`notebooks/har_adaptation_ablation.ipynb`. Sele varianti, con **stessi semi, stessi dati, stesso ottimizzatore**: cambia soltanto la funzione obiettivo. In totale 5 semi × 10 soggetti target × 6 varianti = 300 esecuzioni.

| variante | $\gamma$ | peso $w_i$ | che cosa isola |
|---|---|---|---|
| (a) nessun adattamento | — | — (modello sorgente congelato) | il punto di partenza da battere |
| (b) sola entropia, senza termine di diversità | 0 | 1 | controllo negativo: deve collassare |
| (c) obiettivo completo, senza pesatura | 0.5 | 1 | quanto vale il solo termine di diversità |
| (d) pesatura con l'incertezza del modello puntuale | 0.5 | $\exp(-H^{\text{puntuale}}_i)$ | quanto vale pesare, senza Bayes |
| (e) pesatura con l'incertezza totale bayesiana (fedele al paper) | 0.5 | $\exp(-H^{\text{totale}}_i)$ | il metodo del paper |
| (f) pesatura con la sola componente epistemica (**deviazione**) | 0.5 | $\exp(-z_i)$, con $z_i$ standardizzato su riferimento sorgente fisso | se l'epistemica da sola basti |

Il confronto fra (c), (d) ed (e) è la parte informativa della tabella: (c) dice quanto si ottiene senza pesare, (d) quanto si ottiene pesando con un'incertezza non bayesiana, (e) quanto ne aggiunge il trattamento bayesiano. La variante (f) è dichiaratamente una **deviazione** dal paper, aggiunta per verificare un'intuizione plausibile ma non ovvia: se l'incertezza epistemica è quella che segnala i dati «mai visti», forse pesare con essa sola funzionerebbe meglio che pesare con l'incertezza totale.

**Risultati macro-mediati sui 10 target, media ± dev. std sui 5 semi** (cella 10):

| variante | accuratezza | Δacc vs (a) | ECE | NLL | collasso >90% una classe | **classe azzerata (collasso parziale)** |
|---|---|---|---|---|---|---|
| (a) | 0.9157 ± 0.0022 | 0.0000 | 0.0713 | 0.3710 | 0.0% | 0.0% |
| (b) | 0.5143 ± 0.0287 | −0.4014 | 0.4857 | **13.3756** | **34.0%** | 80.0% |
| (c) | 0.8667 ± 0.0231 | −0.0490 | 0.1333 | 3.6017 | 0.0% | 14.0% |
| (d) | 0.8845 ± 0.0482 | −0.0312 | 0.1155 | 3.1154 | 0.0% | 22.0% |
| **(e)** | **0.8945 ± 0.0401** | **−0.0212** | **0.1055** | **2.8405** | 0.0% | **24.0%** |
| (f) | 0.8779 ± 0.0385 | −0.0378 | 0.1220 | 3.2541 | 0.0% | 30.0% |

> **Che cos'è il collasso, e come si misura.** «Collasso» indica qui il fenomeno anticipato in §3.2: l'obiettivo che chiede predizioni nette trova la scorciatoia di assegnare tutti i dati alla stessa classe. Nella tabella è misurato in due modi diversi, e la distinzione è essenziale. La colonna «collasso totale» conta le esecuzioni in cui più del 90% delle predizioni finisce su una sola classe: è il collasso vistoso, quello che nessuno può non notare. La colonna «classe azzerata» conta le esecuzioni in cui **almeno una classe non riceve nemmeno una predizione**, anche se le altre restano ben distribuite: è un collasso **parziale**, molto meno visibile e altrettanto grave in un'applicazione reale, dove significa che una delle attività da riconoscere è semplicemente sparita dal repertorio del sistema.

Il controllo negativo, variante (b), fa quello che deve: collassa completamente nel 34% delle esecuzioni, con verosimiglianza logaritmica negativa attorno a 13.4 — il valore esploso che, come spiegato sopra, segnala predizioni sicure e sbagliate. L'ordinamento dele varianti pesate è (e) > (d) > (f) > (c), coerente col paper: la pesatura bayesiana fedele al paper è la migliore, quella con l'incertezza del solo modello puntuale la segue, l'IM non pesato è il peggiore dei quattro.

Ma **macro-mediato su tutti i 10 target, ogni variante di adattamento è peggiore del non fare niente**, incluso (e). Non è un bug: 6 dei 10 target sono già ≥94% (§6.1), quindi la media è dominata da «l'adattamento sposta un modello già corretto dal suo ottimo» e nasconde «l'adattamento salva un soggetto genuinamente spostato» come effetto di minoranza.

**Scomposizione per terzili di spostamento** (cella 17).

> **Che cos'è un terzile e perché si usa qui.** I **terzili** dividono un insieme ordinato in tre parti di ampiezza confrontabile: si ordinano i 10 soggetti target per indicatore di spostamento e si guarda separatamente il terzo meno spostato, quello intermedio e quello più spostato. Il motivo per farlo è enunciato nella tesi del paper stessa: il vantaggio atteso non è uniforme, si manifesta *dove lo spostamento è forte*. Una media su tutti i soggetti mescola due regimi opposti — quelli dove non c'è niente da guadagnare e quelli dove c'è — e la maggioranza numerica del primo gruppo può cancellare l'effetto del secondo. Guardare per terzili non è un espediente per far apparire un risultato: è la scomposizione che la tesi da verificare prescrive **prima** di guardare i numeri.

Terzili per indicatore di spostamento: basso `27, 15, 21`, medio `7, 30, 11`, alto `22, 20, 8, 19`. I due terzili basso e medio coincidono esattamente con i 6 soggetti già al 94% o più senza adattamento; il terzile alto contiene i 4 genuinamente difficili (dal 68% al 92%).

| variante | Δacc tercile basso | Δacc tercile medio | Δacc tercile **alto** |
|---|---|---|---|
| (b) | −0.2374 ± 0.0522 | −0.5100 ± 0.1220 | −0.4429 ± 0.0279 |
| (c) | −0.0339 ± 0.0516 | −0.1237 ± 0.1061 | −0.0043 ± 0.0387 |
| (d) | −0.0178 ± 0.0233 | −0.1302 ± 0.0374 | **+0.0331 ± 0.1031** |
| **(e)** | −0.0164 ± 0.0225 | −0.1325 ± 0.0400 | **+0.0586 ± 0.1024** |
| (f) | −0.0043 ± 0.0048 | −0.0809 ± 0.0576 | −0.0306 ± 0.0656 |

**(d) ed (e) sono gli unicle varianti che diventano positivi, e solo sul tercile alto** — esattamente i 4 soggetti che avevano bisogno di adattamento. È la validazione qualitativa della tesi del paper su dati reali: la vista macro-mediata da sola è fuorviante, la vista per tercile è quella che valida il metodo.

**Va però detto con la stessa chiarezza che l'effetto non è statisticamente distinguibile da zero.** Il +0.0586 della variante (e) sul terzile alto ha una deviazione standard fra semi di ±0.1024: l'intervallo che ne risulta comprende ampiamente lo zero. Lo stesso vale per il +0.0331 della variante (d) (±0.1031).

> **Come si legge questo confronto.** La deviazione standard fra semi misura quanto il risultato oscilla ripetendo l'esperimento con inizializzazioni diverse. Quando quell'oscillazione è più grande dell'effetto medio misurato, l'esperimento non permette di escludere che il vero effetto sia nullo, o perfino di segno opposto: si osserva un miglioramento medio, ma non si può affermare che sia reale. Non è la stessa cosa che dire che l'effetto non esista — l'ordinamento fra le varianti è consistente e va nella direzione prevista su entrambi i terzili in cui è misurabile — ma la differenza fra «indicazione» e «dimostrazione» va enunciata, non lasciata al lettore da dedurre dalle barre d'errore. Su cinque semi e quattro soggetti, quello che si può affermare è che il *segno* dell'effetto è quello previsto e che l'ordinamento fra varianti è consistente, non che l'effetto sia significativo. Il tercile medio, curiosamente, è il più penalizzato per ogni variante (peggio del basso): i soggetti `7, 30, 11` stanno al 94–99.5% con confini di decisione presumibilmente più stretti di `27, 15, 21` (praticamente al 100%), quindi toccare l'estrattore costa loro di più — un'osservazione riportata nella cella markdown 18 del notebook.

La variante (f) resta negativo perfino sul tercile alto (−0.0306): standardizzare la sola epistemica, senza la componente aleatoria dell'entropia totale, non basta a identificare in modo affidabile dove l'adattamento sia da fidarsi. La deviazione dal paper sottoperforma la variante fedele proprio dove conta. Nelle traiettorie i pesi di (f) risultano anche visibilmente più rumorosi.

**La riserva del collasso parziale, riportato con lo stesso peso con cui è stato scoperto.** La colonna «collasso >90% una classe» legge 0% per tutte le varianti tranne (b), e sarebbe facile fermarsi lì. Ma quella soglia cattura solo il collasso *totale*. Una metrica distinta — «qualche classe riceve zero predizioni» — mostra collasso **parziale** nel **14% delle esecuzioni di (c), 22% di (d), 24% di (e), 30% di (f)**. Il caso documentato: sul soggetto 19, la variante (f) assegna zero predizioni a WALKING_DOWNSTAIRS in tutti i 5 semi. Il punto che conta è che **non è specifico di (f)**: la minimizzazione di entropia con testa congelata può azzerare una classe su *qualsiasi* variante con $\gamma > 0$, con frequenza che cresce passando da (c) verso (f). Il 24% di (e) è una riserva reale sui suoi numeri aggregati, che sono i migliori della tabella: la variante "vincente" azzera una classe in circa un'esecuzione su quattro. Questa metrica è stata aggiunta in revisione, dopo che la prima lettura della tabella (che guardava solo la soglia >90%) suggeriva erroneamente che soltanto (b) collassasse.

*Figure: `notebooks/har_adaptation_ablation.ipynb`, cella 17 — sezione "7. Per-shift-tercile breakdown", barre di Δaccuratezza per terzile e variante con barre d'errore sui 5 semi. Cella 15 — sezione "6. Per-arm trajectories", quattro pannelli (funzione obiettivo, termine di entropia pesato, termine di diversità, peso medio per campione) sui 300 passi per il soggetto a shift più alto, banda media ± dev. std sui 5 semi. Cella 13 — sezione "5. Reliability diagrams by arm", sei pannelli, predizioni aggregate su 5 semi × 10 target.*

**Nota sulla riproducibilità di questa sezione.** Tutti i numeri sopra provengono dagli output salvati del notebook come sta su disco (celle 10 e 17), e coincidono con quelli citati in `TODO.md` §9. Va però segnalato che il modello sorgente **non è bit-identico fra esecuzioni successive** dello stesso notebook con gli stessi seed: l'early stopping può cadere a un'epoca diversa per effetto di non-determinismo in virgola mobile, e siccome ogni variante parte da quel modello, i tassi di collasso e i Δ per tercile ereditano questa variabilità *oltre* a quella fra semi già riportata. Questa variabilità **fra esecuzioni** non è quantificata in questo progetto: il protocollo a 5 semi × 10 soggetti stima la variabilità dovuta all'inizializzazione, non quella dovuta a ri-esecuzione. È una ragione ulteriore per leggere il +0.0586 del tercile alto come indicazione di segno e non come stima puntuale.

---

## 7. Estensione a insieme aperto: classi mai viste in addestramento

### 7.1 Che cosa si è cercato di testare

> **Che cos'è l'impostazione a insieme aperto.** Fino a qui il problema era a **insieme chiuso** (in inglese *closed-set*): le classi presenti nei dati target sono le stesse su cui il modello è stato addestrato, e cambia solo la loro distribuzione. Nell'impostazione a **insieme aperto** (in inglese *open-set*) il target contiene **anche classi che il modello non ha mai visto**: nel gergo del campo si chiamano *target-private* («private del target») oppure, più in generale, dati **fuori distribuzione** (in inglese *out-of-distribution*). Il compito non è più solo classificare correttamente, ma anche **accorgersi** che un dato non appartiene a nessuna delle classi conosciute — e farlo senza aver mai visto in addestramento un solo esempio di che aspetto abbia una classe sconosciuta. L'incertezza epistemica sembra il candidato naturale: se significa «non ho dati qui», dovrebbe accendersi proprio su questi casi.

Il paper rivendica vantaggi anche in questa impostazione. Il progetto la costruisce su **HAPT** (*Human Activities and Postural Transitions*, «attività umane e transizioni posturali»), estensione naturale di UCI HAR: stessi 30 soggetti, stesse 561 caratteristiche, più 6 classi di **transizione posturale** — il passaggio dallo stare in piedi allo stare seduti, da seduti a in piedi, da seduti a coricati, e i loro inversi. Sono le fasi di movimento *fra* due posture stabili, che nel dataset originale di UCI HAR erano semplicemente scartate.

Verifica del dataset (`har_openset.ipynb`, celle 4–5): 10 929 finestre, 561 caratteristiche, 30 soggetti (ID 1–30, coincidenti con UCI HAR, verificato con `assert`), 0 NaN/Inf, valori in $[-1,1]$; conteggi delle transizioni 70/33/107/85/139/84.

L'impostazione sperimentale: come dati **appartenenti al dominio noto** si usano le attività di locomozione dei 10 soggetti target di UCI HAR (1 475 finestre); come dati **da rilevare** le transizioni posturali **degli stessi soggetti** prese da HAPT (160 finestre, da 11 a 25 per soggetto). Il modello è il classificatore a 3 classi di §6.1, ricostruito su 5 semi di addestramento (42, 123, 456, 789, 1011), con $M = 5000$ estrazioni. Nessun riaddestramento: si chiede al modello esistente di segnalare da sé i dati che non gli competono.

> **Perché gli stessi soggetti.** Usare le transizioni degli stessi 10 soggetti, e non di altri, elimina un fattore di confusione: se le transizioni venissero da persone diverse, un'eventuale incertezza elevata potrebbe essere dovuta al cambio di persona anziché al cambio di attività, e non si potrebbe distinguere fra le due cause.

### 7.2 Il risultato sorprendente

Si calcola l'area sotto la curva ROC (§5.4) usando l'incertezza come punteggio per distinguere i due gruppi. Un rilevatore che funziona deve stare **sopra** 0.5; 0.5 significa inutile; **sotto** 0.5 significa sistematicamente al contrario (cella 10):

| punteggio usato | area sotto la curva (media ± dev. std su 5 semi) | rapporto fra incertezza sui dati nuovi e sui dati noti |
|---|---|---|
| epistemica | **0.0938 ± 0.0246** | 0.325 ± 0.061 |
| aleatoria | 0.0888 ± 0.0196 | 0.170 ± 0.010 |
| totale | 0.0899 ± 0.0215 | 0.205 ± 0.007 |

L'AUROC non è basso: è **invertito**, e stabilmente (deviazione standard ≤ 0.025 su 5 semi). Il modello è **più certo** sulle transizioni posturali che sulle attività di locomozione che dovrebbe conoscere: l'incertezza epistemica suli dati fuori distribuzione è appena il 32.5% di quella sulle appartenente al dominio noto. Un valore di 0.09 su 5 semi non è rumore né un errore di segno nel calcolo (il punteggio è coerentemente negato per tutte e tre le quantità, e tutte e tre concordano).

### 7.3 La diagnosi: due ipotesi e un controllo controfattuale

> **Come funziona un controllo controfattuale.** Davanti a un risultato inatteso, la tentazione è cercarne *una* spiegazione plausibile e fermarsi. Il procedimento corretto è formulare **almeno due** spiegazioni concorrenti, chiedersi in che cosa farebbero previsioni **diverse**, e costruire un esperimento in cui quelle previsioni si separano. «Controfattuale» significa che l'esperimento modifica **una sola** cosa rispetto all'originale, tenendo tutto il resto identico, per rispondere alla domanda: che cosa sarebbe accaduto se quella sola cosa fosse stata diversa? Se le due ipotesi predicono esiti diversi per quell'unica modifica, l'esito osservato ne scarta una.

Il notebook formula due ipotesi alternative (celle 8 e 10):

1. **Limite di interpolazione dell'approssimazione di Laplace.** Le transizioni posturali, nello spazio delle 561 caratteristiche standardizzate sulla sorgente, cadono **dentro** la regione occupata dai dati sorgente, non fuori: sono movimenti brevi e a bassa energia, statisticamente più vicini al centro della distribuzione di quanto lo siano le camminate di un soggetto con andatura atipica.

   > **Interpolazione ed estrapolazione, e perché la distinzione è decisiva.** **Estrapolare** significa dare una risposta in una regione **fuori** da quella coperta dai dati visti; **interpolare** significa darla **dentro**, fra dati noti. L'approssimazione di Laplace rileva bene l'estrapolazione: lontano dai dati le diverse ipotesi plausibili sui pesi divergono fra loro, il disaccordo cresce, e l'incertezza epistemica si accende. Ma per come è costruita **non può** rilevare una novità che sta *dentro*: se una classe mai vista produce caratteristiche simili a quelle di classi note, le ipotesi sui pesi in quel punto sono tutte d'accordo — proprio perché là ci sono molti dati — e il modello è, correttamente secondo il proprio criterio, sicuro di sé. Il criterio è «quanto sono lontano dai dati», non «questa cosa è una classe che conosco». Con l'estrattore di caratteristiche congelato, nulla nel meccanismo può colmare questa differenza.

2. **Riferimento confondente.** Il gruppo usato come «dominio noto» sono i soggetti *target*, che a loro volta sono spostati rispetto alla sorgente. Il segnale di novità potrebbe esserci ma restare mascherato: se lo spostamento fra persone alza l'incertezza sul gruppo di riferimento, il confronto fra i due gruppi risulta appiattito o rovesciato per un motivo che non ha nulla a che vedere con le classi nuove.

Le due ipotesi fanno previsioni **diverse e distinguibili**, e il notebook esegue il controllo che le separa: ricalcolare la stessa area sotto la curva usando come dominio noto la **verifica interna sorgente** — stesso modello, stessa distribuzione sui pesi, stesso gruppo di dati da rilevare, cambia soltanto il riferimento. Se valesse l'ipotesi 2, rimuovere il fattore di confusione dovrebbe far risalire il valore sopra 0.5; se vale l'ipotesi 1, dovrebbe restare rovesciato.

| gruppo usato come dominio noto | area sotto la curva, epistemica | rapporto fra incertezza sulle transizioni e sul gruppo di riferimento |
|---|---|---|
| attività note dei soggetti **target** | 0.0938 ± 0.0246 | 0.325 ± 0.061 |
| **verifica interna sorgente** | **0.0799 ± 0.0171** | **1.062 ± 0.142** |

**L'ipotesi 1 è confermata, la 2 esclusa.** Con il riferimento pulito l'AUROC resta invertito (0.0799), quindi il confondente non spiegava il risultato. E il rapporto racconta la storia in modo ancora più diretto: l'epistemica media sulle transizioni è **1.06×** quella del verifica interna sorgente, cioè **statisticamente indistinguibile da dati che il modello ha effettivamente visto in addestramento** — mentre le stesse attività di locomozione dei soggetti target arrivano fino a 8.9× (§6.3). Il modello tratta una classe che non ha mai visto come se fosse dominio noto. La logica di questo confronto è codificata nel notebook stesso (cella 10), che stampa «Both references show inverted AUROC (<0.5) → This supports Hypothesis (1): Laplace interpolation limit. HAPT transitions fall within feature space convex hull.»

### 7.4 Perché è un contributo e non solo un risultato negativo

Il valore di questa sezione non è l'AUROC di 0.09 in sé: è che il progetto ha **isolato e diagnosticato un limite reale del metodo con un controllo pulito**. La differenza fra "l'esperimento open-set non ha funzionato" e quanto sopra è la struttura del ragionamento: due ipotesi concorrenti con previsioni distinguibili, un controllo controfattuale che cambia una sola variabile, una conclusione che ne segue e una che viene esclusa, il tutto stabile su 5 semi.

La conseguenza sostanziale per il metodo è precisa e non era ovvia a priori: **l'incertezza epistemica da Laplace sull'ultimo strato è un rilevatore di estrapolazione, non di novità semantica.** Sui benchmark del paper (Office-Home, VisDA-C) le classi target-private sono categorie visive diverse, che una ResNet-50 mappa presumibilmente fuori dal supporto sorgente, e il meccanismo funziona. Con feature pre-ingegnerizzate e classi nuove che sono *interpolazioni* di quelle note, non funziona — e non può, per come è costruito. È un limite del tipo di rappresentazione, non dell'implementazione.

**Due lacune di questa sezione, da dichiarare.** Il piano di lavoro (`TODO.md` §10) marca come completati due punti che **non sono presenti nel notebook eseguito**.

Il primo sono le due metriche standard dell'impostazione a insieme aperto, usate dal paper (§4 di U-SFAN, seguendo il protocollo di SHOT).

> **Come funzionano queste due metriche.** Nell'impostazione a insieme aperto il sistema deve poter rispondere anche «non è nessuna delle classi che conosco», il che richiede di aggiungere una classe artificiale chiamata *sconosciuta* e una soglia sull'incertezza oltre la quale assegnarvi un dato. L'accuratezza calcolata su tutte le classi, **compresa** quella sconosciuta, si indica in letteratura con la sigla OS (*open-set accuracy*); quella calcolata sulle sole classi effettivamente condivise fra sorgente e target si indica con OS\* (*known-class accuracy*). Servono entrambe, e insieme: un sistema che dichiarasse tutto «sconosciuto» avrebbe un'ottima accuratezza sulla classe sconosciuta e un'accuratezza pessima sulle classi note, mentre uno che non dichiarasse mai nulla sconosciuto avrebbe l'inverso. Il notebook realizza invece uno studio di **rilevamento** basato sull'area sotto la curva, che valuta l'ordinamento delle incertezze senza mai fissare una soglia: è una cosa affine ma diversa, e non produce quelle due accuratezze.

Il secondo punto mancante è la variante di ripiego prevista dal piano, che rimuoveva la classe LAYING (stare coricati) dalle sei classi della sorgente mantenendola fra i dati target, per costruire una classe mai vista in un modo alternativo. Inoltre le dimensioni campionarie OOD sono piccole (160 finestre totali, 11–25 per soggetto), ai limiti della soglia di 10 che il notebook stesso si dà.

`[DA DISCUTERE INSIEME: se aggiungere OS/OS* con una soglia sull'incertezza (breve da fare, riusando l'artefatto già salvato in data/har_bald_artifact_openset.npz), o se ridichiarare l'ambito di §10 come "rilevamento OOD" e correggere di conseguenza le spunte in TODO.md. Data la diagnosi di §7.3, un'accuratezza OS basata su soglia sull'incertezza sarebbe prevedibilmente vicina al caso — che è comunque un risultato riportabile e coerente.]`

---

## 8. Limitazioni

### 8.1 Obsolescenza della distribuzione sui pesi durante l'adattamento

È il limite concettualmente più profondo, ed è **strutturale al metodo del paper**, non un difetto di questa implementazione. Il titolo usa il termine inglese *staleness*, che si può rendere con **obsolescenza**: la distribuzione sui pesi diventa via via vecchia rispetto alla rete a cui si riferisce.

> **Come nasce il problema.** La distribuzione sui pesi della testa è calcolata una volta sola, sommando i contributi delle caratteristiche prodotte dall'estrattore **così com'era al termine dell'addestramento sorgente**. Durante l'adattamento, però, l'estrattore è l'unica cosa che cambia: dopo cento passi produce caratteristiche diverse da quelle su cui la distribuzione era stata tarata. È come se un perito stimasse il margine di errore di uno strumento e poi lo strumento venisse progressivamente modificato, continuando a usare la stima originale. La curvatura da cui l'incertezza è ricavata dipende dalle caratteristiche, quindi dopo $k$ passi la distribuzione corretta sarebbe quella dell'estrattore aggiornato, non quella iniziale.

L'implementazione mitiga in parte il problema — `laplace.predictive(model, X_target, ...)` è richiamata a **ogni passo** e valuta le caratteristiche *correnti*, quindi i pesi non sono congelati al passo 0 — ma $\theta_{\text{MAP}}$ e $H^{-1}$ restano quelli di $\beta_{\text{MAP}}$. Si campionano teste plausibili per un estrattore che non c'è più. Dopo 300 passi con lr $10^{-2}$ la deriva può essere sostanziale, e l'entità della staleness **non è misurata** in questo progetto. Un'estensione naturale sarebbe rifittare la Laplace periodicamente durante l'adattamento — ma sarebbe una deviazione dal paper, che fissa esplicitamente la posterior (Fig. 3b: «we keep the posterior over the parameters fixed»), e richiederebbe i dati sorgente, che nell'impostazione source-free non ci sono. È quindi un limite intrinseco al setting, che vale la pena enunciare come tale.

### 8.2 Natura locale e a un solo picco dell'approssimazione, mostrata empiricamente

Questo limite è di solito solo citato; qui è **visibile nei dati**, ed è il motivo per cui il test MCMC di §5.2 è più informativo di un semplice "✓ passato".

Il problema di test è **quasi separabile**: 60 punti in tre nuvole gaussiane ben distanziati, con accuratezza MAP 0.983 e test 1.000. In questo regime la log-verosimiglianza è quasi piatta lungo le direzioni che aumentano il margine — spingere i pesi più in là non peggiora l'adattamento ai dati, e solo la prior li trattiene. La posterior vera è quindi **asimmetrica** lungo quelle direzioni, con moda diversa dalla media, mentre la Laplace è per costruzione una gaussiana **simmetrica centrata sulla moda**.

I marginali campionati con MH lo mostrano direttamente. I valori sono stampati dentro la figura, componente per componente:

| componente | MCMC $\mu$ | MCMC $\sigma$ | Laplace $\mu$ | Laplace $\sigma$ |
|---|---|---|---|---|
| $\theta[0]$ | 0.817 | 0.945 | **0.299** | 0.821 |
| $\theta[2]$ | −0.526 | 0.763 | −0.623 | 0.808 |
| $\theta[4]$ | −0.419 | 0.942 | −0.364 | 0.854 |
| $\theta[6]$ | 0.315 | 0.704 | **−0.034** | 0.831 |
| $\theta[8]$ | −0.560 | 0.837 | −0.509 | 0.880 |
| $\theta[11]$ | 0.873 | 1.260 | 0.831 | 1.096 |

Su $\theta[0]$ e $\theta[6]$ la moda MAP è spostata rispetto alla media della posterior vera di circa 0.35–0.52, cioè **circa metà deviazione standard a posteriori**, e l'istogramma MCMC è visibilmente asimmetrico (coda destra pesante) mentre la gaussiana rossa è simmetrica sopra la moda. Su altre componenti ($\theta[2]$, $\theta[8]$) l'accordo è buono. Il difetto non è uniforme: è selettivo sulle direzioni in cui la quasi-separabilità rende la likelihood piatta.

*Figura: `notebooks/test_laplace_vs_mcmc.ipynb`, cella 12 — sezione "Compare Posterior Weight Distributions", sei pannelli con istogramma MCMC (blu) e densità gaussiana Laplace (rosso) sovrapposti, e le statistiche μ/σ dei due in un box per pannello.*

La conseguenza a valle è la distorsione sistematica di §5.2: tutte le incertezze predittive della Laplace stanno **sopra** la diagonale rispetto a MCMC, con scostamento medio di 0.29–0.32 nat su totale e aleatoria. Da qui una conclusione operativa importante per come leggere §6: **l'ordinamento fornito dall'epistemica è validato, il suo livello assoluto no.** Il progetto, per fortuna, usa l'epistemica solo come ranking (Spearman, z-score su quantili sorgente fissi) e mai come valore assoluto interpretato — la validazione copre l'uso che se ne fa. Ma questo va detto esplicitamente, non lasciato implicito.

**Un riserva sulla riserva, per onestà:** il riferimento MCMC è una catena Metropolis–Hastings a random walk di 5 000 campioni (dopo 1 000 di burn-in) in 12 dimensioni, con tasso di accettazione 0.299. Non sono stati calcolati diagnostici di convergenza ($\hat{R}$, ESS, autocorrelazioni) né sono state confrontate catene multiple da inizializzazioni diverse. Una catena mal mescolata sotto-stima la dispersione della posterior e potrebbe produrre marginali apparentemente asimmetrici: parte della discrepanza osservata potrebbe quindi essere dovuta al riferimento e non alla Laplace. L'asimmetria osservata è **coerente** con la quasi-separabilità, ma un test più rigoroso richiederebbe HMC/NUTS con diagnostici.

`[DA DISCUTERE INSIEME: quanto forte tenere questa affermazione. Le tre opzioni sono (i) tenerla come sopra, con il caveat esplicito sulla catena; (ii) rafforzarla aggiungendo diagnostici di convergenza al notebook di test — poche righe di codice, ma è un rerun; (iii) indebolirla a "compatibile con", rinunciando al claim "dimostrato empiricamente". Io propendo per (ii) se c'è tempo, altrimenti (i).]`

### 8.3 Il cambio di dataset, da elettromiografia a sensore inerziale

Il progetto era stato concepito su un dataset sEMG di arto inferiore (BASAN), con un'ipotesi clinicamente motivata: addestrare su soggetti sani e adattare a soggetti con patologia diagnosticata al ginocchio, dove la biomeccanica alterata costituisce uno shift di dominio genuino e non arbitrario (`docs/project_overview_it.md`). Il dataset è stato accantonato per problemi di qualità del segnale grezzo, il cui recupero avrebbe richiesto un lavoro di elaborazione del segnale (segmentazione, filtraggio, rimozione di artefatti, estrazione di feature) sproporzionato rispetto al focus **metodologico** del corso — che riguarda l'inferenza bayesiana, non la pulizia di segnali biomedici.

Ciò che si è perso in questa scelta va detto chiaramente, perché non è marginale:

- **Lo shift è cambiato di natura.** Sano vs patologico è uno shift con un meccanismo causale noto e una severità potenzialmente graduabile clinicamente; la variabilità inter-soggetto su HAR è più mite (rapporto 1.20× rispetto al riferimento intra-sorgente, §4.3) e priva di una variabile di severità indipendente. Il proxy a distanza fra centroidi è un surrogato grezzo, come mostra il disaccordo sul soggetto 20 (§6.4).
- **Il regime sperimentale è diventato quello sfavorevole per il metodo.** Il paper mostra i suoi vantaggi maggiori sotto shift *forte*; su HAR 6 target su 10 sono già ≥94% e non c'è nulla da adattare. Questa è la ragione strutturale per cui l'effetto di §6.5 è piccolo e non significativo — non un difetto di implementazione, ma una conseguenza diretta della scelta di dataset.
- **La motivazione applicativa si è indebolita.** L'argomento assistivo (un dispositivo indossabile che non deve essere sicuro quando sbaglia) resta valido come inquadramento, ma HAR non lo mette alla prova.

Il pivot è stata la scelta giusta rispetto agli obiettivi del corso — un pipeline bayesiano completo, validato e onestamente riportato su un dataset pulito vale più di un pipeline incompleto su un dataset problematico — ma va contabilizzato come un ridimensionamento della domanda di ricerca, non come una sostituzione neutra.

### 8.4 Il limite di interpolazione (§7)

L'incertezza epistemica da Laplace sull'ultimo strato **non rileva classi nuove che cadono dentro l'inviluppo delle caratteristiche sorgente**: AUROC 0.08–0.09 su 5 semi, con l'epistemica suli dati fuori distribuzione pari a 1.06× quella del verifica interna sorgente. Diagnosticato con controllo controfattuale (§7.3). Limita il metodo al rilevamento di **estrapolazione**, che è ciò che serve nell'impostazione closed-set con shift covariato, ma non nell'open-set con classi semanticamente nuove ma statisticamente interne. Su feature pre-ingegnerizzate come le 561 di HAR/HAPT questo limite è particolarmente stretto, perché la rappresentazione non è appresa per separare classi che non erano nel task sorgente.

### 8.5 Il collasso parziale nel 14–30% delle esecuzioni (§6.5)

La metrica di collasso ovvia (>90% delle predizioni su una classe) legge 0% per tutte le varianti pesate e nasconde il problema. La metrica più stretta (qualche classe con zero predizioni) scatta sul 14% delle esecuzioni di (c), 22% di (d), **24% di (e)** e 30% di (f). Il variante fedele al paper, quello con i migliori numeri aggregati, azzera una classe in circa un'esecuzione su quattro. La minimizzazione di entropia con testa congelata può azzerare una classe su qualsiasi variante con $\gamma > 0$; il termine di diversità $\mathcal{L}_{\text{div}}$ a $\gamma = 0.5$ evita il collasso *totale* ma non quello parziale. In un contesto assistivo, un classificatore di modalità locomotoria che smette del tutto di prevedere "discesa di scale" sarebbe un fallimento operativo, non una degradazione graduale — quindi questa riserva pesa più di quanto suggerisca la sua entità sull'accuratezza aggregata.

A questo si aggiunge il fatto che questi tassi sono stimati su 50 esecuzioni per variante (5 semi × 10 soggetti) e con un modello sorgente che non è bit-identico fra ri-esecuzioni del notebook (§6.5): la loro incertezza è quindi maggiore di quanto suggerisca una singola cifra percentuale. Il protocollo non è abbastanza potente per stimarli con precisione, e la conclusione robusta è ordinale — il collasso parziale è presente su tutte le varianti con $\gamma > 0$ e cresce da (c) verso (f) — non la singola percentuale.

### 8.6 Popolazione e sensore di UCI HAR

I 30 soggetti sono **giovani adulti sani** (19–48 anni), con smartphone montato **alla vita** in condizioni di laboratorio. Le implicazioni:

- **Nessuna patologia, nessuna alterazione dell'andatura**: la variabilità inter-soggetto qui è variazione normale, non biomeccanica alterata. Il ridimensionamento rispetto all'ipotesi originale è quello discusso in §8.3.
- **Non è un esoscheletro.** Il nome del progetto (*BayesianExoAdaptation*) descrive l'inquadramento motivazionale, non il setup sperimentale: un dispositivo di assistenza al movimento avrebbe sensori multipli su segmenti corporei, dinamica di accoppiamento con l'utente, e vincoli di latenza. Qui la classificazione è **offline**, su finestre di 2.56 s già estratte.
- **Feature pre-ingegnerizzate e non riapprese.** Le 561 caratteristiche sono fisse; non c'è apprendimento di rappresentazione dai segnali inerziali grezzi (che il dataset contiene, e che `TODO.md` §12 elencava come possibile estensione). Come mostra §7, la rigidità della rappresentazione è direttamente responsabile del limite di interpolazione.
- **Nessuna asimmetria di costo.** Tutti gli errori pesano uguale nelle metriche riportate, mentre in un contesto assistivo confondere "discesa di scale" con "camminata in piano" ha conseguenze diverse dal confonderle nell'ordine opposto. La curva accuratezza-copertura di §6.4 è il passo più vicino a un trattamento asimmetrico (astensione invece di risposta), ma non è una vera analisi di costi.

### 8.7 L'approssimazione fattorizzata dell'Hessiana non è stata implementata

Scelta deliberata e motivata: con una testa di 195 parametri l'Hessiana esatta si inverte direttamente, e la fattorizzazione secondo Kronecker introdurrebbe un'ipotesi di indipendenza (la somma $\sum_n \Lambda_n \otimes \phi_n\phi_n^\top$ non si scompone in un prodotto) per ottenere un'approssimazione peggiore di una soluzione esatta già disponibile, senza alcun vantaggio computazionale dimostrabile a questa scala (§4.1). La conseguenza è che il progetto **non** ha una misura dell'errore introdotto da KFAC (varianza predittiva, ECE, correlazione di ranking BALD) rispetto all'esatta, che era l'ultimo item di `TODO.md` §6 e che sarebbe stato di per sé interessante. Se un revisore volesse quel confronto, richiederebbe una rete con testa molto più larga per essere significativo.

### 8.8 Altre limitazioni minori ma dichiarate

- **Criterio di convergenza MC non soddisfatto** (§6.2): $M = 5000$ è scelto come miglior valore disponibile, non come valore convergente secondo il criterio dichiarato. Il rumore MC residuo sulle stime epistemiche è dell'ordine dell'1% relativo.
- **Convenzione su $\tau_{\text{prior}}$**: implementata come $\text{wd}\times N_{\text{train}} = 25.10$ anziché $\text{wd}\times N_{\text{source}} = 31.97$, mentre l'Hessiana somma su tutte le 3 197 finestre sorgente (§4.5).
- **Due temperature non coincidenti nell'adattamento**: i logit della funzione obiettivo usano $\tau = 0.4$, la predittiva che genera i pesi usa il default 1.0 (§4.4). Non è chiaro se sia intenzionale.
- **Split sorgente/target singolo**: i 5 semi variano solo l'inizializzazione del modello, non la partizione dei soggetti. La LOSO completa sui 30 soggetti, prevista come opzione in `TODO.md` §1, non è stata eseguita, quindi tutti i risultati sono condizionati a una particolare assegnazione 20/10.
- **Un solo soggetto per il tracciamento delle traiettorie** (il 19, a shift massimo): le traiettorie di §6.5 non sono rappresentative dell'intera coorte.
- **Percorso assoluto locale** in un output salvato di `har_openset.ipynb` (cella 13), lasciato deliberatamente (`REFACTOR_NOTES.md`, finding 5).

---

## 9. Conclusione

La catena di ragionamento del progetto, dall'inizio alla fine, con i numeri chiave e senza sovrastimarli.

**Lo strumento è corretto.** L'Hessiana esatta a $K$ classi si riduce alla formula binaria delle dispense §7.4 entro $5.7\times10^{-6}$ in assoluto, con la struttura a blocchi attesa (§5.1). La predittiva concorda con Metropolis–Hastings sulla posterior vera di un problema a 12 parametri con correlazioni di Pearson 0.94 (totale), 0.95 (aleatoria), 0.72 (epistemica) — con una distorsione sistematica di livello di ~0.3 nat che rende validato l'**ordinamento** ma non la scala assoluta (§5.2, §8.2). La covarianza è definita positiva ed esattamente simmetrica su tutti i problemi testati.

**Il meccanismo funziona dove lo shift è forte e inequivocabile.** Sul problema giocattolo bidimensionale, sotto shift forte, l'adattamento non pesato non recupera nulla (0.682 → 0.684; sulla classe spostata 6.0% → 7.3%) mentre l'adattamento guidato dall'incertezza porta la classe spostata da 6.0% a 98.7% — con il controllo negativo `mild` che resta a 1.000 (§5.3, in un regime di iperparametri scelto perché il fallimento fosse visibile, e dichiarato come tale). Su MNIST vs Fashion-MNIST è specificamente la componente **epistemica** a portare il segnale: rapporto 10.9×, AUROC 0.987 contro 0.946 dell'entropia MAP grezza e 0.979 dell'entropia totale (§5.4). La decomposizione non è un ornamento: separa un segnale utile da uno che lo diluisce.

**Su uno shift naturale mite l'incertezza sa di cosa parla, e produce un effetto reale ma statisticamente più piccolo.** Su UCI HAR l'epistemica traccia l'indicatore di spostamento con **ρ = 0.84 ± 0.03 su 5 semi, tutti p < 0.01** (§6.4); il soggetto più difficile della coorte è anche quello con l'epistemica più alta, a 8.9× il riferimento sorgente, e su una scala assoluta fissata dai quantili sorgente i difficili stanno a $z > 17$ contro $z \lesssim 2.5$ dei facili (§6.3). I controlli di significato passano nella direzione prescritta dalla teoria (lontano dai dati → epistemico; SITTING/STANDING → aleatorio, elevazione 5.7× contro 2.5×). La Laplace migliora ECE e NLL su 7 soggetti su 11, con i guadagni concentrati sui difficili (soggetto 20: NLL 1.55 → 1.03) e una lieve sovracorrezione sui già ben calibrati (§6.4). Sull'adattamento, le varianti con pesatura da incertezza sono gli **unici** che diventano positivi, e **solo** sul tercile di shift alto (+0.06 per la variante fedele al paper, +0.03 per quello con incertezza MAP) — ma con deviazione standard fra semi (±0.10) che comprende lo zero, e con un collasso parziale di classe nel 24% delle esecuzioni della variante migliore (§6.5). L'affermazione sostenibile è: **il segno e l'ordinamento dell'effetto sono quelli previsti dal paper, la sua significatività statistica su questi dati no.**

**E c'è un limite riconosciuto e diagnosticato.** L'estensione open-set su HAPT produce un AUROC invertito (0.09, stabile su 5 semi): l'incertezza epistemica non vede affatto le transizioni posturali come classi nuove. Il controllo controfattuale con il verifica interna sorgente come riferimento esclude l'ipotesi del riferimento confondente (AUROC resta 0.08) e conferma quella del **limite di interpolazione**: l'epistemica suli dati fuori distribuzione è 1.06× quella dei dati di addestramento, cioè indistinguibile da dominio noto (§7). La Laplace sull'ultimo strato è un rilevatore di **estrapolazione**, non di novità semantica interpolata. Questo non è un fallimento dell'esperimento, è la sua conclusione: un limite reale del metodo, isolato con un controllo che cambia una sola variabile.

Il filo che tiene insieme i quattro blocchi è quello annunciato nel piano originale del progetto, e ogni anello regge o cede in modo documentato: c'è davvero uno shift da studiare (sì, ma mite in media e molto eterogeneo, §4.3); il punto di partenza è affidabile (sì, 0.975 su verifica interna sorgente, con intervallo 0.653–1.000 sui target, §6.1); l'incertezza del modello sa di cosa parla (sì, ρ = 0.84, con validazione MCMC dell'ordinamento e non del livello, §6.4); quella conoscenza si traduce in un adattamento migliore quando conta (**direzionalmente sì, ma non in modo statisticamente distinguibile su questi dati, e con una riserva di collasso parziale nel 24% delle esecuzioni**, §6.5). E dove il meccanismo non può funzionare, il progetto lo ha mostrato e spiegato invece di ometterlo (§7).

---

## 10. Riferimenti

1. **Roy, S., Trapp, M., Pilzer, A., Kannala, J., Sebe, N., Ricci, E., Solin, A.** — *Uncertainty-guided Source-free Domain Adaptation*. European Conference on Computer Vision (ECCV), 2022. arXiv:2208.07591. — Paper replicato; copia locale in `paper/U-SFAN.pdf`. Riferimenti puntuali usati: §3 (definizione del problema, $f = h\circ g$, testa congelata in adattamento), Eq. 1–3 (SHOT-IM: cross-entropy con label smoothing, $\mathcal{L}_{\text{ent}}$, $\mathcal{L}_{\text{div}}$), Eq. 4–5 (posterior predittiva, Laplace sull'ultimo strato), Eq. 6 (integrazione MC), Eq. 7 ($\mathcal{L}^{\text{ug}}_{\text{ent}}$ con $w_i = \exp(-H)$), Fig. 2 (natura mode-seeking della Laplace; rilevamento di dati fuori distribuzione), Fig. 3b (posterior fissata durante l'adattamento), Fig. 4 (problema giocattolo bidimensionale, shift mite vs forte, confine di decisione ribaltato), §4 (protocollo open-set, metriche OS/OS\*).

2. **Kristiadi, A., Hein, M., Hennig, P.** — *Being Bayesian, Even Just a Bit, Fixes Overconfidence in ReLU Networks*. International Conference on Machine Learning (ICML), 2020, pp. 5436–5446. — Riferimento [26] del paper U-SFAN; base della Laplace sull'ultimo strato e dell'argomento sull'overconfidence delle reti ReLU. Replicato in scala ridotta in `notebooks/mnist_check.ipynb` (§5.4).

3. **Liang, J., Hu, D., Feng, J.** — *Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation* (SHOT). ICML, 2020. — Riferimento [35] del paper U-SFAN; origine dell'obiettivo IM usato come base (variante (c) dell'ablazione) e del protocollo di valutazione open-set.

4. **Dispense del corso** — *Probabilistic Machine Learning*, Università degli Studi di Trieste (`PML_notes_full.pdf`, non versionato nel repository). Sezioni usate: §2.3 (entropia, divergenza KL, informazione mutua), §7.3 (approssimazione di Laplace), §7.4 (regressione logistica bayesiana, formula $S_N^{-1} = S_0^{-1} + \sum_n s_n(1-s_n)\phi_n\phi_n^\top$), §10.6 (Bayesian model averaging), cap. 8 (metodi Monte Carlo a catena di Markov).

5. **Anguita, D., Ghio, A., Oneto, L., Parra, X., Reyes-Ortiz, J. L.** — *A Public Domain Dataset for Human Activity Recognition Using Smartphones*. ESANN, 2013. — **UCI HAR**, UCI Machine Learning Repository n. 240: 30 soggetti, smartphone alla vita, 561 caratteristiche per finestra di 2.56 s, 10 299 finestre. Dati non versionati nel repository (si veda `README.md`).

6. **Reyes-Ortiz, J. L., Oneto, L., Samà, A., Parra, X., Anguita, D.** — *Transition-Aware Human Activity Recognition Using Smartphones*. Neurocomputing, 2016. — **HAPT** (*Smartphone-Based Recognition of Human Activities and Postural Transitions*), UCI Machine Learning Repository n. 341: estensione di UCI HAR con 6 classi di transizione posturale, usate come classi target-private nell'esperimento open-set (§7).

---

## Appendice A — Glossario delle sigle e dei termini inglesi

Raccolti in un unico posto tutti i termini tecnici usati nel report, con la sezione in cui sono spiegati per esteso. Le sigle sono elencate per come compaiono in letteratura, perché è la forma in cui il lettore le incontrerà altrove.

### Sigle

| sigla | forma completa | in italiano | dove è spiegata |
|---|---|---|---|
| AUROC | *area under the receiver operating characteristic curve* | area sotto la curva caratteristica operativa | §5.4 |
| BALD | *Bayesian Active Learning by Disagreement* | decomposizione dell'incertezza per disaccordo fra ipotesi | §3.2 |
| ECE | *expected calibration error* | errore di calibrazione atteso | §6.4 |
| HAPT | *Human Activities and Postural Transitions* | dataset di attività umane e transizioni posturali | §7.1 |
| HAR | *Human Activity Recognition* | dataset di riconoscimento dell'attività umana | §4.2 |
| IM | *information maximization* | massimizzazione dell'informazione | §2.1, §3.2 |
| IQR | *interquartile range* | scarto interquartile | §6.3 |
| KFAC | *Kronecker-factored approximate curvature* | curvatura approssimata fattorizzata secondo Kronecker | §4.1 |
| KL | divergenza di *Kullback–Leibler* | divergenza di Kullback–Leibler | §3.2 |
| LOSO | *leave-one-subject-out* | lascia fuori un soggetto per volta | §4.3 |
| MAP | *maximum a posteriori* | stima a massima probabilità a posteriori, o stima puntuale | §3.1 |
| MCMC | *Markov chain Monte Carlo* | catene di Markov Monte Carlo | §3.4 |
| MLP | *multi-layer perceptron* | rete neurale completamente connessa | §4.1 |
| NLL | *negative log-likelihood* | verosimiglianza logaritmica negativa | §6.4 |
| OOD | *out-of-distribution* | fuori distribuzione | §5.4, §7.1 |
| OS, OS\* | *open-set accuracy*, *known-class accuracy* | accuratezza a insieme aperto, accuratezza sulle classi note | §7.4 |
| PCA | *principal component analysis* | analisi delle componenti principali | §4.3 |
| ReLU | *rectified linear unit* | unità lineare rettificata | §2.1 |
| SFDA | *source-free domain adaptation* | adattamento di dominio senza accesso ai dati sorgente | §2.1 |
| SHOT | *Source HypOthesis Transfer* | trasferimento dell'ipotesi sorgente (metodo di riferimento) | §4.4 |
| U-SFAN | *Uncertainty-guided Source-free domain AdaptatioN* | nome del metodo del paper replicato | §2.2 |

### Termini inglesi

| termine | significato in questo report | dove è spiegato |
|---|---|---|
| *ablation study* | studio comparativo che rimuove un componente per volta per misurarne il contributo | §6.5 |
| *arm* (variante) | ciascuna delle varianti messe a confronto in uno studio comparativo | §6.5 |
| *batch* | il gruppo di dati elaborati insieme in un passo di calcolo | §6.3 |
| *burn-in* | fase iniziale di rodaggio di una catena di campionamento, da scartare | §3.4 |
| *closed-set* / *open-set* | impostazione a insieme chiuso (stesse classi) o aperto (classi nuove nel target) | §7.1 |
| *coverage* (copertura) | frazione di casi su cui il sistema scelga di rispondere invece di astenersi | §6.4 |
| *domain shift* | spostamento di dominio: la popolazione dei dati cambia fra addestramento e uso | §2.1 |
| *early stopping* | arresto anticipato dell'addestramento quando la verifica smette di migliorare | §6.1 |
| *feature* | caratteristica: uno dei numeri che descrivono un dato | §4.1 |
| *feature extractor* | estrattore di caratteristiche: la parte iniziale della rete | §4.1 |
| *head* (testa) | l'ultimo strato lineare della rete, quello trattato in modo bayesiano | §4.1 |
| *learning rate* | passo di apprendimento dell'ottimizzatore | §6.1 |
| *notebook* | documento eseguibile che alterna testo, codice e risultati | §1 |
| *overconfidence* | eccesso di confidenza: dichiararsi certi più di quanto i dati giustifichino | §2.1 |
| *reliability diagram* | diagramma di affidabilità: confronto fra confidenza dichiarata e accuratezza reale | §6.4 |
| *seed* (seme) | numero iniziale del generatore casuale, che rende ripetibile un esperimento | §1 |
| *selective prediction* | predizione selettiva: rispondere solo quando si è abbastanza certi | §6.4 |
| *softmax* | trasformazione che converte punteggi grezzi in probabilità che sommano a uno | §2.1 |
| *source-private*, *target-private* | classi presenti solo nella sorgente o solo nel target | §7.1 |
| *staleness* | obsolescenza: una stima che si riferisce a una versione superata del modello | §8.1 |
| *sweep* | scansione: esperimento in cui si fa variare per gradi un solo parametro | §5.3 |
| *toy problem* | problema giocattolo, costruito piccolo per rendere ispezionabile il meccanismo | §5.3 |
| *weight decay* | decadimento dei pesi: penalizzazione sui pesi grandi durante l'addestramento | §3.1, §4.5 |

### Le due componenti dell'incertezza, in una riga ciascuna

- **Incertezza epistemica** — nasce dal fatto che i dati non sono bastati a determinare i pesi del modello. Ipotesi diverse, tutte plausibili, danno risposte diverse. Si ridurrebbe con più dati. È quella che dovrebbe accendersi lontano dai dati di addestramento (§3.2).
- **Incertezza aleatoria** — nasce dal fatto che il dato è intrinsecamente ambiguo: due classi si sovrappongono davvero in quella regione. Non si riduce con più dati, perché non è ignoranza del modello ma rumore del problema (§3.2).

---

## Appendice B — Indice dei notebook e delle sezioni di `TODO.md`

| notebook | sezione `TODO.md` | contenuto | sezione di questo report |
|---|---|---|---|
| `loader.ipynb` | §2 | caricamento, validazione, split per soggetto, indicatore di spostamento, PCA | §4.2–4.3 |
| `test_hessian_binary.ipynb` | §1b.1, §5 | validazione della riduzione $K=2$ | §5.1 |
| `test_laplace_vs_mcmc.ipynb` | §1b.2, §5 | validazione contro Metropolis–Hastings | §5.2, §8.2 |
| `synthetic_track.ipynb` | §3 | replica Fig. 4, controllo di significato, sweep di shift | §5.3 |
| `mnist_check.ipynb` | §3b | MNIST vs Fashion-MNIST, Rotated-MNIST | §5.4 |
| `har_source_training.ipynb` | §4 | addestramento a stima puntuale, valutazione per soggetto, fit della Laplace | §6.1 |
| `har_mc_convergence.ipynb` | §5 | convergenza MC, generazione dell'artefatto BALD | §6.2 |
| `har_uncertainty_decomposition.ipynb` | §7 | decomposizione BALD, controlli di significato, normalizzazione, sweep di $\tau$ | §6.3 |
| `har_calibration.ipynb` | §8 | affidabilità, ECE/NLL/Brier, AUROC, copertura, Spearman | §6.4 |
| `har_adaptation_ablation.ipynb` | §9 | 6 varianti, terzili di shift, traiettorie | §6.5 |
| `har_openset.ipynb` | §10 | rilevamento di dati fuori distribuzione su HAPT, controllo controfattuale | §7 |
| — | §6 (approssimazione fattorizzata) | **deliberatamente non implementata** | §4.1, §8.7 |

I due notebook di validazione dell'Hessiana si trovano nella cartella `notebooks/`, non in una cartella `tests/` separata (che nel repository non esiste).

## Appendice C — Punti aperti raccolti

Riepilogo dei `[DA DISCUTERE INSIEME]` sparsi nel testo, in ordine di impatto sul report:

1. **§8.2** — quanto tenere forte l'affermazione sulla non-gaussianità della posterior, dato che la catena MH non ha diagnostici di convergenza.
2. **§7.4** — se aggiungere le metriche OS/OS\* e il fallback LAYING (marcati fatti in `TODO.md` §10 ma assenti dal notebook) o ridichiarare l'ambito di §10.
3. **§5.3** — se la prima versione non monotona della sweep può essere raccontata come risultato osservato, dato che non è conservata in git.
4. **§4.5** — se rieseguire con $\tau_{\text{prior}} = 0.01 \times 3197$ come controllo, o dichiarare la convenzione implementata.
5. **§6.4** — gli estremi esatti della curva accuratezza-copertura per il soggetto 19 sono letti dalla figura, non stampati; se serve un numero, va aggiunta una `print`.
6. **§6.5 / §8.5** — se quantificare la variabilità fra ri-esecuzioni (rieseguire l'ablazione *n* volte e riportare la dispersione dei tassi di collasso), oppure lasciarla dichiarata ma non misurata come sta ora.

**Risolti dopo la ri-esecuzione dei notebook:** l'ambiguità su quale esecuzione dell'ablazione fosse canonica (l'esecuzione divergente non esiste più; quella su disco è l'unica) e l'output mancante della tabella Rotated-MNIST in `mnist_check.ipynb`, ora presente e verificabile.

**Da valutare prima della consegna:** 4 degli 11 notebook (`har_adaptation_ablation`, `har_openset`, `loader`, `test_laplace_vs_mcmc`) hanno `execution_count` non sequenziali, cioè gli output salvati provengono da una sessione in cui altre celle erano già state eseguite, non da un'esecuzione pulito dall'alto in basso. I numeri sono coerenti con tutto il resto e riproducibili, ma un revisore che aprisse quei notebook non vedrebbe una traccia di esecuzione lineare. `loader` e `test_laplace_vs_mcmc` costano pochi secondi da rieseguire; gli altri due sono più lenti.
