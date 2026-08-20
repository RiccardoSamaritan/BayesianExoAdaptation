# Incertezza epistemica come guida all'adattamento di dominio source-free

### Replica ed estensione di Roy et al., *Uncertainty-guided Source-free Domain Adaptation* (U-SFAN, ECCV 2022) su riconoscimento di modalità locomotoria

**Corso:** Probabilistic Machine Learning — Università degli Studi di Trieste
**Codice:** `github.com/RiccardoSamaritan/BayesianExoAdaptation`

---

## Abstract

Un classificatore puntuale addestrato su una popolazione e applicato a un'altra tende a sbagliare *con sicurezza*: le reti ReLU estrapolano in modo confidente lontano dai dati visti, e un obiettivo di adattamento non supervisionato che inseguisse ciecamente quelle predizioni può ribaltare il confine di decisione nella direzione sbagliata. Questo progetto replica il meccanismo proposto da Roy et al. — approssimazione di Laplace esatta sull'ultimo strato, decomposizione BALD dell'incertezza predittiva, pesatura per campione dell'obiettivo di *information maximization* — e lo verifica in tre stadi di difficoltà crescente: un toy 2D dove il meccanismo è visibile, MNIST vs Fashion-MNIST dove lo shift è inequivocabile (rapporto epistemico 10.9x, AUROC OOD 0.987 contro 0.946 dell'entropia MAP), e infine UCI HAR dove lo shift è naturale e mite (variabilità inter-soggetto). Su HAR l'incertezza epistemica traccia il proxy di shift con $\rho$ di Spearman = 0.84 ± 0.03 su 5 seed (tutti p < 0.01) e la Laplace migliora ECE/NLL su 7 soggetti su 11, con i guadagni concentrati sui più difficili; l'effetto dell'adattamento guidato è però positivo **solo** nel tercile di shift alto (+0.06 accuratezza), con deviazione standard tra seed (±0.10) che comprende lo zero, e con un collasso parziale di classe nel 14–30% dei run. Un'estensione open-set su HAPT produce un AUROC *invertito* (0.09) stabile su 5 seed: un controllo controfattuale con la source-validation come riferimento isola la causa nel limite di interpolazione della Laplace sull'ultimo strato. Il risultato complessivo è che il meccanismo funziona dove lo shift è forte, produce un effetto reale ma statisticamente debole dove lo shift è mite, e ha un limite riconosciuto e diagnosticato.

---

## 1. Nota metodologica sulla verificabilità

Ogni affermazione quantitativa in questo report è seguita dal riferimento al notebook e alla cella che l'ha prodotta, nella forma `(nome_notebook.ipynb, cella N)`, dove `N` è l'indice della cella nel file `.ipynb` (0-based, come nel JSON del notebook). Le figure **non** sono duplicate su disco: esistono come output inline nei notebook, e per ognuna è indicato dove trovarla. Dove un'interpretazione richiesta non è documentata nei notebook, è marcata `[DA DISCUTERE INSIEME]` invece di essere inferita.

I riferimenti alle dispense del corso (`PML_notes_full.pdf`) usano la numerazione di sezione citata in `README.md` e `TODO.md` §11; il PDF delle dispense non è versionato nel repository, quindi i numeri di sezione non sono verificabili dal repo stesso.

---

## 2. Introduzione e motivazione

### 2.1 Il problema

Nella *source-free domain adaptation* (SFDA) si dispone di un modello addestrato su un dominio sorgente etichettato e di dati non etichettati di un dominio target, ma **non** dei dati sorgente. L'allineamento delle distribuzioni per confronto diretto è quindi impossibile: l'unica supervisione disponibile è il modello stesso.

Il fulcro del problema è che un modello puntuale non ha alcuna nozione di quanto i dati abbiano vincolato i suoi pesi. Le reti con attivazioni ReLU estrapolano linearmente fuori dal supporto dei dati di addestramento e producono softmax arbitrariamente confidenti in regioni mai viste (Hein et al., citato come [23] nel paper; Kristiadi, Hein & Hennig, ICML 2020, come [26]). Un obiettivo di adattamento non supervisionato come l'*information maximization* — che premia predizioni a bassa entropia — amplifica questa patologia: dove il modello sorgente è confidentemente sbagliato, l'IM rinforza l'errore.

### 2.2 Il paper replicato

Roy et al., *Uncertainty-guided Source-free Domain Adaptation* (U-SFAN, arXiv:2208.07591, ECCV 2022) propone una soluzione volutamente leggera. La tesi centrale, in tre passi:

1. Al modello sorgente si aggiunge, **post hoc e in un singolo passaggio forward**, un trattamento bayesiano del solo ultimo strato (*last-layer Laplace approximation*, §3.2 del paper, Eq. 5), lasciando l'estrattore di feature deterministico.
2. Da quella posterior si ricava, via integrazione Monte Carlo (Eq. 6), una distribuzione predittiva e da essa un peso per campione $w_i = \exp(-H_i)$, con $H_i$ l'entropia della media predittiva.
3. Il peso modula il solo termine di entropia condizionale dell'obiettivo IM (Eq. 7): i punti target su cui il modello ammette di non sapere influiscono meno sull'aggiornamento. Il termine di diversità di batch resta non pesato.

Il vantaggio rivendicato non è tanto accuratezza media quanto **robustezza a shift forti** — precisamente il regime in cui SHOT-IM convenzionale fallisce (Fig. 4b del paper: «IM, when used with a MAP estimate, finds a completely flipped decision boundary»).

### 2.3 Perché è un progetto di Probabilistic Machine Learning

Il progetto non usa il vocabolario bayesiano come decorazione: ogni suo passaggio è un oggetto visto a lezione.

- Il trattamento bayesiano dell'ultimo strato **è** regressione logistica bayesiana multiclasse con approssimazione di Laplace (dispense §7.3–7.4), applicata su feature apprese $\phi(x) = g_\beta(x)$ invece che scelte a mano. L'Hessiana implementata è la generalizzazione a $K$ classi della formula delle dispense, e la riduzione a $K=2$ è verificata numericamente (§5.1 di questo report).
- La decomposizione dell'incertezza è **esattamente** l'informazione mutua fra predizione e parametri (dispense §2.3), stessa forma funzionale che compare — su un oggetto diverso — nell'obiettivo di adattamento (§4.4).
- La predizione è **Bayesian model averaging** (dispense §10.6): non una predizione, ma la media su un ventaglio di ipotesi pesate dalla posterior.
- Il campionamento MCMC (dispense cap. 8) è usato non come metodo di inferenza ma come **verità di riferimento** per validare l'approssimazione, su un problema abbastanza piccolo perché il confronto sia possibile.

La domanda sperimentale è quindi una domanda probabilistica: *l'incertezza sui parametri stimata da un'approssimazione locale sa effettivamente di cosa parla, e quella conoscenza è utile?*

---

## 3. Background teorico

### 3.1 Approssimazione di Laplace e regressione logistica bayesiana (dispense §7.3–7.4)

Sia $f = h \circ g$, con $g$ estrattore di feature (parametri $\beta$) e $h$ testa lineare (parametri $\theta$). Fissato $\beta = \beta_{\text{MAP}}$, la log-posterior sulla sola testa è

$$\log p(\theta \mid \mathcal{D}) = \sum_n \log \varphi_{y_n}\!\big(\theta^\top \phi_n\big) - \tfrac{1}{2}\tau \lVert\theta\rVert^2 + \text{cost.}, \qquad \phi_n = [\,g_{\beta_{\text{MAP}}}(x_n);\,1\,]$$

cioè **letteralmente** la log-posterior di una regressione logistica multiclasse con prior gaussiana isotropa di precisione $\tau$, con la sola differenza che le feature $\phi_n$ sono apprese anziché progettate. L'approssimazione di Laplace è lo sviluppo di Taylor al secondo ordine attorno al massimo:

$$p(\theta \mid \mathcal{D}) \approx \mathcal{N}\!\big(\theta \mid \theta_{\text{MAP}}, H^{-1}\big), \qquad H = -\nabla^2_\theta \log p(\theta\mid\mathcal{D})\big|_{\theta_{\text{MAP}}}$$

Per la log-likelihood softmax l'Hessiana esatta (Gauss–Newton generalizzata) è

$$H = \sum_{n=1}^{N} \Lambda_n \otimes \phi_n \phi_n^\top + \tau I, \qquad \Lambda_n = \operatorname{diag}(p_n) - p_n p_n^\top$$

con $p_n$ il softmax MAP nel punto $n$. Nel caso $K=2$ si ha $\Lambda_n = s_n(1-s_n)\begin{bmatrix}1 & -1\\ -1 & 1\end{bmatrix}$, e ogni blocco diagonale di $H$ si riduce a $S_N^{-1} = S_0^{-1} + \sum_n s_n(1-s_n)\phi_n\phi_n^\top$, la formula binaria delle dispense §7.4. Questa riduzione è verificata numericamente (§5.1).

L'implementazione è in `src/bayesian.py` (`LastLayerLaplace.fit`), completamente vettorizzata via `einsum`, con simmetrizzazione `cov = 0.5*(cov + cov.T)` dopo l'inversione per rimuovere l'asimmetria numerica.

Il carattere **locale e unimodale** dell'approssimazione è una proprietà da tenere presente, non un dettaglio: la Laplace cattura la curvatura attorno a *un* modo e nulla del resto della posterior (Fig. 2a del paper). Il §8.2 mostra empiricamente dove questo si vede.

### 3.2 Entropia, KL, informazione mutua (dispense §2.3): BALD e IM sono la stessa forma funzionale

Data una distribuzione predittiva ottenuta mediando su un insieme di ipotesi, la decomposizione di riferimento è

$$\underbrace{\mathbb{H}\!\left[\mathbb{E}_{\theta}\, p(y\mid x,\theta)\right]}_{\text{totale}} \;=\; \underbrace{\mathbb{E}_{\theta}\!\left[\mathbb{H}\!\left[p(y\mid x,\theta)\right]\right]}_{\text{aleatoria}} \;+\; \underbrace{\mathbb{I}\!\left[y;\theta \mid x\right]}_{\text{epistemica (BALD)}}$$

L'epistemica è dunque l'**informazione mutua** fra la predizione e i parametri: è grande quando ipotesi diverse, tutte plausibili sotto la posterior, sono in disaccordo su quel punto. È l'incertezza che si ridurrebbe con più dati. L'aleatoria è ciò che resterebbe anche conoscendo esattamente i pesi: sovrapposizione intrinseca fra classi.

L'obiettivo di adattamento ha **esattamente la stessa forma funzionale su un oggetto diverso**. L'IM loss (paper Eq. 2–3, implementata in `src/im_adapt.py::im_loss`) è

$$\mathcal{L}_{\text{IM}}^{\text{(da massimizzare)}} = \underbrace{\mathbb{H}\!\left[\tfrac{1}{N}\textstyle\sum_i p(y\mid x_i)\right]}_{\text{entropia della media}} - \underbrace{\tfrac{1}{N}\textstyle\sum_i \mathbb{H}\!\left[p(y\mid x_i)\right]}_{\text{media delle entropie}} \;=\; \mathbb{I}\!\left[y;x\right]_{\hat{p}_{\text{emp}}}$$

cioè l'informazione mutua fra input e predizione sotto la distribuzione empirica del target. La media è **sui dati** invece che **sui pesi**. Il termine di diversità del paper, $\mathcal{L}_{\text{div}} = D_{\mathrm{KL}}(\hat p \,\|\, K^{-1}\mathbf{1}) - \log K = -\mathbb{H}[\hat p]$, è la stessa entropia della media riscritta come divergenza KL dall'uniforme (identità verificata nel docstring di `src/im_adapt.py`).

Questa coincidenza di forma è didatticamente il punto più elegante del progetto: la stessa quantità informazionale serve, su un asse, a *misurare* quanto il modello non sa (BALD, media sui pesi) e, sull'altro, a *ottimizzare* la struttura delle predizioni sul target (IM, media sui dati).

Un'avvertenza di implementazione, che vale ribadire perché è facile confonderla: il peso che entra nella loss è $w_i = \exp(-H_i^{\text{totale}})$, cioè l'entropia **totale**, non la sola componente epistemica (paper Eq. 7). La decomposizione epistemica/aleatoria è una *diagnostica* del perché $H_i$ sia grande; la variante che pesa con la sola epistemica è una deviazione deliberata testata come braccio (f) dell'ablazione (§6.5).

### 3.3 Bayesian model averaging (dispense §10.6)

La predizione non è $\varphi(h_{\theta_{\text{MAP}}}(z))$ ma la media sulla posterior (paper Eq. 5–6), approssimata per MC:

$$p(y_k \mid z, \mathcal{D}) \approx \frac{1}{M}\sum_{j=1}^{M} \varphi_k\!\big(h_{\theta_j}(z)/\tau\big), \qquad \theta_j \sim \mathcal{N}(\theta_{\text{MAP}}, H^{-1})$$

Il campionamento avviene via fattore di Cholesky della covarianza (`LastLayerLaplace.sample_heads`). Il parametro $\tau$ (`temperature`) divide i logit *prima* del softmax, per ogni campione. Il numero $M$ di campioni MC non è un dettaglio implementativo ma un iperparametro con un criterio di scelta esplicito (§6.2).

### 3.4 MCMC come verità di riferimento (dispense cap. 8)

L'approssimazione di Laplace è, per costruzione, un'approssimazione: prima di fidarsene serve confrontarla con la posterior vera su un problema abbastanza piccolo perché quest'ultima sia campionabile. `notebooks/test_laplace_vs_mcmc.ipynb` fa esattamente questo con Metropolis–Hastings a random walk sulla log-posterior esatta della testa (12 parametri: $K=3$, $D_p=4$). È il collaudo dello strumento, non un metodo alternativo di inferenza: nel pipeline reale la posterior serve migliaia di volte durante l'adattamento e MCMC sarebbe proibitivo. I risultati, incluse le discrepanze, sono in §5.2 e §8.2.

---

## 4. Metodologia

### 4.1 Architettura e dimensione dell'Hessiana

MLP `561 → 128 → 64 → 3`: estrattore $g$ = due strati lineari con ReLU, testa $h$ = un `Linear(64, 3)`. Conteggi verificati: 80 387 parametri totali, di cui 80 192 nell'estrattore e **195 nella testa** (`har_source_training.ipynb`, cella 13). La posterior è quindi su $K\cdot(D+1) = 3 \times 65 = 195$ parametri, e $H$ è una matrice $195 \times 195$: covarianza confermata definita positiva, autovalori in $[3.10\times10^{-4},\, 3.98\times10^{-2}]$, simmetria esatta (`max |cov - cov^T| = 0.00e+00`) (`har_source_training.ipynb`, cella 23).

**Perché §6 (KFAC) è stato deliberatamente saltato.** Il paper originale ricorre alla Kronecker-factored Laplace approximation perché lavora con ResNet-50 e molte classi, dove anche la sola Hessiana dell'ultimo strato può essere ingestibile. Qui la matrice è $195 \times 195$: si inverte esattamente in millisecondi. Implementare KFAC significherebbe introdurre un'ipotesi di indipendenza (la somma $\sum_n \Lambda_n \otimes \phi_n\phi_n^\top$ **non** fattorizza in generale come prodotto di Kronecker) per ottenere un'approssimazione peggiore della soluzione esatta già disponibile, senza alcun vantaggio computazionale da dimostrare a questa scala. La scelta è documentata nel docstring di `src/bayesian.py` («This is exact (no KFAC) because the feature dimension keeps K·(D+1) small enough to invert directly») e in `TODO.md` §6. È un'omissione motivata, non un item incompleto: la conseguenza è che ogni risultato di questo report usa l'Hessiana **esatta**, non un'approssimazione di essa.

### 4.2 Dataset e il pivot da BASAN

Il progetto nasceva su un dataset sEMG di arto inferiore (BASAN), con l'idea di addestrare su soggetti sani e adattare a soggetti con patologia al ginocchio (`docs/project_overview_it.md`). Il dataset è stato accantonato per problemi di qualità del segnale grezzo che avrebbero richiesto un lavoro di data engineering sproporzionato rispetto al focus metodologico del corso; l'implementazione corrente usa **UCI HAR** (30 soggetti, smartphone alla vita, 561 feature pre-ingegnerizzate per finestra di 2.56 s). Il framework bayesiano, la domanda di ricerca e il piano di validazione sono rimasti quelli originali. Il dettaglio e le conseguenze sono in §8.3.

Validazione del caricamento: 10 299 finestre × 561 feature, 30 soggetti, 0 NaN, 0 Inf, tutti i valori in $[-1,1]$, 0 valori fuori range (`loader.ipynb`, cella 6). Il task principale è la locomozione a 3 classi (WALKING / WALKING_UPSTAIRS / WALKING_DOWNSTAIRS), 4 672 finestre (`loader.ipynb`, cella 11).

### 4.3 Split per soggetto e proxy di shift

Lo split originale train/test del dataset è scartato; la partizione è **per soggetto**, con seed 42: pool sorgente di 20 soggetti (`1,2,3,4,5,6,9,10,12,13,14,16,17,18,23,24,25,26,28,29`, 3 197 finestre) e 10 soggetti target (`7,8,11,15,19,20,21,22,27,30`, 1 475 finestre) (`loader.ipynb`, cella 11). Lo `StandardScaler` è fittato **solo** sul pool sorgente e applicato ai target.

Il proxy di shift per soggetto target è la distanza euclidea fra il centroide del soggetto e la media del pool sorgente, nello spazio standardizzato a 561 dimensioni (`loader.ipynb`, cella 13):

| soggetto | 27 | 15 | 21 | 7 | 30 | 11 | 22 | 20 | 8 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|
| distanza | 7.41 | 9.67 | 10.03 | 10.16 | 11.06 | 11.41 | 18.19 | 18.43 | 19.89 | 29.21 |

Media 14.54, dev. std 6.39. **Il proxy va preso per quello che è**: il riferimento a shift quasi-nullo — distanze leave-one-subject-out *dentro* il pool sorgente — vale in media 12.15 ± 5.00, quindi il rapporto target/sorgente è solo **1.20×**, sotto la soglia di 2× che il notebook stesso si era dato; il notebook stampa infatti «⚠ Target shifts are comparable to source baseline - minimal shift detected» (`loader.ipynb`, cella 15). Il controllo di sanità (split casuale di finestre ignorando il soggetto) dà 0.95, cioè praticamente zero, confermando che il riferimento è costruito bene. La conclusione onesta è che **su HAR lo shift medio è mite**; ciò che il resto del progetto sfrutta è la sua forte eterogeneità (7.41 a 29.21), non la sua entità media. Questo è determinante per interpretare l'ablazione (§6.5).

*Figura: `notebooks/loader.ipynb`, cella 19 — sezione "9. Visualization: Source vs Target in PCA Space" (due pannelli: tutte le finestre sorgente vs target, e centroidi per soggetto etichettati). PCA fittata solo sul sorgente, varianza spiegata 28.6% + 9.0% = 37.65% (cella 17).*

### 4.4 Protocollo di adattamento

Adattamento source-free su dati target non etichettati: si aggiorna **solo** $\beta$ (l'estrattore $g$), la testa $h$ resta congelata assieme alla sua posterior (paper §3, Fig. 3b; `src/im_adapt.py::adapt_target`, che chiama `requires_grad_(False)` sui parametri di `model.h`). Iperparametri fissati in `TODO.md` §1 e usati nell'ablazione: $\gamma = 0.5$, $\tau = 0.4$, 300 passi, Adam con lr $10^{-2}$, $M = 100$ campioni MC per passo. I pesi sono **ricalcolati a ogni passo** dalle feature correnti.

Il down-weighting tocca solo il termine di entropia condizionale, mai $\mathcal{L}_{\text{div}}$ (verificato in `im_loss`: `ent_term = (weights * per_sample_ent).mean()`, mentre `div_term = -entropy(p_hat)` non vede i pesi).

Due dettagli implementativi che vale enunciare per precisione: (i) i logit della loss sono divisi per $\tau = 0.4$, mentre la predittiva usata per calcolare i pesi è invocata con la `temperature` di default 1.0 — le due temperature non coincidono; (ii) la chiamata alla predittiva dentro il ciclo non passa un `generator`, quindi il rumore MC varia da passo a passo (l'esperimento resta riproducibile grazie al `torch.manual_seed(seed)` iniziale). Il secondo punto contribuisce alla rumorosità visibile delle traiettorie dei pesi.

**Nessuna etichetta target è mai usata per l'adattamento o per la selezione del modello.** Le etichette target compaiono esclusivamente nel calcolo delle metriche finali riportate. Lo stesso vale per la scelta di $M$, di $\tau$, del numero di passi e dei seed: tutti fissati a priori o scelti su dati sorgente.

### 4.5 Convenzione su $\tau_{\text{prior}}$ e le sue due validazioni indipendenti

`TODO.md` §1b impone $\tau_{\text{prior}} = \text{weight\_decay} \times N$ (convenzione PyTorch: il `weight_decay` è un coefficiente per campione medio, la precisione della prior è la somma). Nel pipeline HAR: $0.01 \times 2510 = 25.10$ (`har_source_training.ipynb`, cella 15).

**Nota di precisione onesta:** il fattore usato è $N_{\text{train}} = 2510$ (il sottoinsieme di training del pool sorgente), mentre l'Hessiana è poi sommata su tutte le 3 197 finestre del pool sorgente completo, e `TODO.md` §1b parla di $N_{\text{source}}$. La discrepanza nasce dal fatto che `tau_prior` è restituito da `train_source_model` (che conosce solo il suo training set) mentre `LastLayerLaplace.fit` è chiamato sul pool intero. Con $\tau = 25.10$ invece di $31.97$ la prior è leggermente più debole del previsto; l'effetto sui risultati è presumibilmente marginale, ma la convenzione dichiarata e quella implementata non sono identiche. `[DA DISCUTERE INSIEME: se vale la pena rieseguire con τ = 0.01 × 3197 come controllo, o dichiararlo come scelta.]`

L'Hessiana esatta è validata in due modi indipendenti prima di costruirci sopra qualsiasi cosa (§5.1 e §5.2).

---

## 5. Validazione della macchina prima dei dati reali

L'ordine di questa sezione è deliberato: nessun numero su HAR è stato interpretato prima che lo strumento fosse collaudato su problemi dove la risposta giusta è nota.

### 5.1 Validazione 1 — riduzione a $K=2$ contro la formula binaria

`notebooks/test_hessian_binary.ipynb`. Problema minimo: $N=20$, $D=3$, $K=2$, estrattore identità (nessuno strato nascosto), $\tau_{\text{prior}} = 0.01 \times 20$. L'Hessiana generale è $8\times8$; si estraggono i blocchi $D_p \times D_p = 4\times4$ e si confrontano con $\sum_n s_n(1-s_n)\phi_n\phi_n^\top + \tau I$ calcolata a mano (cella 10).

| confronto | max diff. assoluta | max diff. relativa | esito |
|---|---|---|---|
| $H_{00}$ vs formula binaria | $5.72\times10^{-6}$ | $1.33\times10^{-6}$ | ✓ |
| $H_{11}$ vs formula binaria | $4.77\times10^{-6}$ | $1.11\times10^{-6}$ | ✓ |
| $H_{01} = -H_{00}$ (senza prior) | $7.15\times10^{-7}$ | — | ✓ |

(`test_hessian_binary.ipynb`, celle 14, 16, 18). La struttura a blocchi riflette la ridondanza della parametrizzazione softmax per $K=2$ (un solo grado di libertà effettivo), e la ritrova correttamente. L'implementazione generale a $K$ classi è quindi verificata sul caso in cui esiste una formula chiusa di riferimento nelle dispense.

### 5.2 Validazione 2 — predittiva contro Metropolis–Hastings

`notebooks/test_laplace_vs_mcmc.ipynb`. Problema: $N_{\text{train}} = 60$, $N_{\text{test}} = 20$, $D = 3$, $K = 3$, tre blob gaussiani ben separati; testa a 12 parametri. MAP: accuratezza train 0.983, test 1.000 (cella 6). MH a random walk sulla log-posterior esatta: 5 000 campioni dopo 1 000 di burn-in, step 0.25, tasso di accettazione 0.299 (cella 10).

Correlazioni di Pearson fra Laplace e MCMC sui 20 punti di test (cella 16):

| quantità | correlazione | diff. assoluta media | diff. assoluta max |
|---|---|---|---|
| entropia totale | 0.939 | 0.324 nat | 0.423 nat |
| epistemica | 0.717 | 0.032 nat | 0.087 nat |
| aleatoria | 0.952 | 0.292 nat | 0.369 nat |
| probabilità medie | — | 0.075 | 0.165 |

Due letture, entrambe importanti e da non confondere.

**L'ordinamento è validato.** Le correlazioni 0.94/0.95 su totale e aleatoria sono alte; l'epistemica è la più debole (0.72), come atteso: è il termine di magnitudine minore ed è una *differenza* fra due stime MC, quindi eredita il rumore di entrambe. Poiché il pipeline reale usa l'epistemica come **segnale di ranking** (correlazione di Spearman con il proxy di shift, z-score su quantili sorgente fissi) e non come valore assoluto, questa è la proprietà che serve.

**Il livello assoluto non lo è.** Nella figura di confronto, *tutti* i punti stanno sopra la diagonale di perfetta corrispondenza, per tutte e tre le quantità: la Laplace **sovrastima sistematicamente** l'incertezza rispetto alla posterior campionata su questo problema, di circa 0.29–0.32 nat in media su totale e aleatoria. Non è rumore, è un offset con un segno.
*Figura: `notebooks/test_laplace_vs_mcmc.ipynb`, cella 18 — sezione "Visual Comparison of Uncertainty Estimates" (tre scatter Laplace vs MCMC con diagonale di riferimento e le correlazioni nei titoli).*

La diagnosi di questo offset — posterior quasi-separabile e non gaussiana — è discussa nelle Limitazioni (§8.2), perché è lì che appartiene concettualmente: non è un bug dell'implementazione, è il carattere locale dell'approssimazione che si manifesta.

### 5.3 Toy sintetico 2D: replica della Fig. 4

`notebooks/synthetic_track.ipynb`. Tre classi 2D (rosso/verde/blu), estrattore MLP `2 → 32 → 16`, MAP con accuratezza train 1.000, covarianza $51\times51$ con autovalori in $[0.0290,\, 22.2234]$ (cella 3). Due regimi di shift: `mild` (tutti i centri leggermente spostati — funge da **controllo negativo**: un target già corretto che l'adattamento non deve rompere) e `strong` (il cluster blu trascinato in una regione che il modello sorgente chiama confidentemente "rosso").

Risultati (cella 7):

| regime | prima dell'adattamento | dopo IM convenzionale | dopo IM guidato dall'incertezza |
|---|---|---|---|
| mild — accuratezza totale | 1.000 | 1.000 | 1.000 |
| mild — solo classe blu | 1.000 | 1.000 | 1.000 |
| **strong** — accuratezza totale | **0.682** | **0.684** | **0.971** |
| **strong** — solo classe blu | **0.060** | **0.073** | **0.987** |

Questa è la replica qualitativa e quantitativa della Fig. 4 del paper: sotto shift forte l'IM convenzionale non recupera nulla (0.682 → 0.684; sulla classe spostata 6.0% → 7.3%), mentre la pesatura per incertezza porta la classe blu da 6.0% a 98.7%. Il controllo negativo `mild` non degrada.

*Figure: `notebooks/synthetic_track.ipynb`, cella 7 — sezione "Box 1 -- Fig. 4 replication", due figure 2×3 (una per regime), righe = MAP convenzionale vs guidato dall'incertezza, colonne = Source / Target / SFDA (IM), con superficie di probabilità sfumata in base alla confidenza. Cella 9: mappe affiancate di incertezza epistemica e aleatoria del modello sorgente, con i due punti sonda marcati.*

**Due calibrazioni specifiche del toy da dichiarare, non da nascondere** (entrambe documentate nelle celle markdown 2 e 6 del notebook):

1. `TAU_SCALE = 0.01`: la prior è scalata a un centesimo della convenzione $\tau = \text{wd}\times N$. Al valore nominale la posterior sull'ultimo strato è così concentrata che la patologia da manuale delle reti ReLU si ripresenta quasi ovunque — il *gap* medio fra i logit cresce allontanandosi dai dati tanto quanto la loro deviazione standard, e la confidenza non scende mai in modo apprezzabile con la distanza. Allargare la posterior rende visibile a distanze finite la correzione in stile Kristiadi et al. È una calibrazione del toy, non una ricetta generale: il pipeline HAR e il check MNIST usano la convenzione nominale senza correzioni.
2. `GAMMA = 0.2` invece di 0.5: a $\gamma = 0.5$ il termine di diversità di batch è già da solo abbastanza forte da far ritrovare il cluster spostato **indipendentemente** dalla pesatura (entrambe le righe convergono alla risposta corretta), il che non dimostra nulla. A $\gamma = 0.2$ il termine di entropia condizionale non pesato domina abbastanza da riprodurre il fallimento del paper. Il rescue mostrato sopra è dunque ottenuto in un regime di iperparametri **scelto perché il fallimento fosse visibile**; non è una previsione su cosa accada a $\gamma = 0.5$ (che è il valore usato nell'ablazione reale, §6.5, dove l'effetto è molto più piccolo).

**Check semantico** (cella 11, con `assert` che lo rendono un test e non una figura da interpretare): un punto lontano da ogni cluster sorgente, $(-4,-4)$, dà epistemica 0.3404 > aleatoria 0.1490 → **epistemico-dominante**; un punto interno al supporto ma sul confine verde/blu, $(1.0,\, 0.0)$, dà aleatoria 0.7904 ≫ epistemica 0.0642 → **aleatorio-dominante**. È il comportamento che la teoria prescrive, verificato su punti scelti geometricamente (il secondo è il punto medio fra due centri sorgente, quindi si documenta da sé).

**Sweep di magnitudine di shift** (cella 13): il cluster blu è trascinato **radialmente verso l'esterno** dal centroide delle tre classi, in 10 livelli di spostamento noto (fino a 3.5). Spearman fra magnitudine di shift e epistemica media sulla classe spostata: **ρ = 0.927, p = 1.12×10⁻⁴**.

*Figura: `notebooks/synthetic_track.ipynb`, cella 13 — sezione "Box 3 -- Shift-magnitude sweep", curva epistemica media vs magnitudine di shift con ρ e p nel titolo.*

La scelta della direzione **radiale** è un punto metodologico, non cosmetico, e la motivazione è documentata nel docstring di `src/toy.py::make_classification_sweep` e nella cella markdown 12: muovendo il cluster verso il centro di un'altra classe la relazione non può essere monotona, perché a un certo punto il cluster spostato si avvicina di nuovo a densità sorgente e l'epistemica *scende* mentre la magnitudine di shift continua a crescere. Muovendolo radialmente verso l'esterno, il cluster si allontana monotonamente da ogni cluster sorgente e la monotonia dell'ipotesi è ben posta. Lo stesso fenomeno si osserva su dati reali con Rotated-MNIST (§5.4): non è un artefatto del toy ma una proprietà generale dell'incertezza epistemica, che misura distanza dai dati visti e non "quantità di corruzione".

`[DA DISCUTERE INSIEME: la versione precedente della sweep, con direzione fissa arbitraria e risultato non monotono, non è conservata nella storia git del repository — src/toy.py contiene la versione radiale sin dal primo commit. Se vogliamo raccontarla come risultato osservato e non solo come motivazione di design, serve dirlo in modo attribuibile (per es. "una prima versione, non conservata, mostrava…") oppure ri-eseguirla come controllo esplicito.]`

### 5.4 MNIST vs Fashion-MNIST e Rotated-MNIST

`notebooks/mnist_check.ipynb`, replica dell'esperimento OOD di Kristiadi, Hein & Hennig (ICML 2020) — riferimento [26] del paper U-SFAN — a una scala eseguibile su CPU. Estrattore: CNN piccola (due blocchi convolutivi + FC, feature a 128 dimensioni), testa `128 → 10`, cioè 1 290 parametri bayesiani. Il codice Laplace/BALD è **identico** a quello del toy: cambia solo $g$. Prior alla convenzione nominale $\tau = \text{wd}\times N$, **senza** la correzione del toy: le feature della CNN non mostrano la stessa patologia allineata ai raggi (cella markdown 4). Covarianza $1290\times1290$ con autovalori in $[9.35\times10^{-6},\, 1.12]$; accuratezza su MNIST test 0.9780 (cella 5).

Confronto MNIST test (2 000 immagini) vs Fashion-MNIST test (2 000 immagini), stesso formato, contenuto completamente diverso (cella 7):

| segnale | MNIST | Fashion-MNIST | rapporto |
|---|---|---|---|
| entropia MAP (nessun Bayes) | 0.1007 | 0.8712 | 8.65× |
| **epistemica (BALD)** | **0.0502** | **0.5471** | **10.91×** |
| aleatoria (BALD) | 0.1356 | 0.7878 | 5.81× |
| entropia totale (Laplace) | 0.1858 | 1.3349 | 7.18× |

Come rilevatore OOD (cella 10, AUROC con MNIST = in-distribution, Fashion-MNIST = OOD):

| score | AUROC |
|---|---|
| entropia MAP (nessun Bayes) | 0.9459 |
| **epistemica (BALD)** | **0.9871** |
| entropia totale (Laplace) | 0.9791 |

Il punto da sottolineare è **quale** componente porta il segnale. Non è "il trattamento bayesiano aiuta" in senso generico: è specificamente la componente **epistemica** a dare il rapporto più grande (10.9×, contro 8.65× dell'entropia MAP e 7.18× della totale) e l'AUROC migliore (0.987 contro 0.946 del MAP grezzo e 0.979 della totale). L'entropia totale, che mescola epistemica e aleatoria, è *peggiore* dell'epistemica pura su questo compito, perché la componente aleatoria risponde all'ambiguità fra classi e non alla novità dell'input. La decomposizione non è un ornamento teorico: separa un segnale utile da uno che lo diluisce.

*Figura: `notebooks/mnist_check.ipynb`, cella 8 — sezione "MNIST test vs. Fashion-MNIST test", tre istogrammi sovrapposti (entropia MAP, epistemica BALD, entropia totale) con le due distribuzioni a confronto. Cella 3: campioni di esempio dei due dataset.*

**Rotated-MNIST** (cella 12; stesso modello e stessa posterior, nessun riaddestramento): shift controllato e continuo, con severità ordinabile (l'angolo).

| angolo | 0° | 20° | 40° | 60° | 80° | 100° | 120° | 140° | 160° | 180° |
|---|---|---|---|---|---|---|---|---|---|---|
| accuratezza | 0.978 | 0.905 | 0.585 | 0.256 | 0.160 | **0.116** | 0.118 | 0.210 | 0.287 | 0.306 |
| epistemica | 0.0502 | 0.1105 | 0.2878 | 0.3833 | 0.4342 | **0.4438** | 0.3908 | 0.3464 | 0.2807 | 0.2575 |
| aleatoria | 0.1356 | 0.3105 | 0.6521 | 0.7784 | 0.7669 | 0.7096 | 0.6889 | 0.6783 | 0.6015 | 0.5301 |

Il pattern è **non monotono per costruzione, non per errore**: l'epistemica cresce ripidamente fino a un picco attorno ai 100° (0.4438, cioè 8.8× il valore a 0°), poi *rientra* verso i 180° (0.2575). La spiegazione, documentata nella cella markdown 11, è che a rotazioni vicine a 180° alcune cifre tornano ad assomigliare a un'altra cifra plausibile (un 6 ruotato somiglia a un 9) anziché a una forma irriconoscibile: il modello torna in una regione dove *ha* visto dati, semplicemente con l'etichetta sbagliata. Lo specchio si vede nell'accuratezza, che ha il minimo a 100–120° (0.116) e risale parzialmente a 0.306 a 180°. È esattamente lo stesso fenomeno che motiva la direzione radiale nella sweep del toy (§5.3): l'epistemica misura distanza dal supporto dei dati, non gravità della corruzione.

*Figura: `notebooks/mnist_check.ipynb`, cella 13 — sezione "Part 2 -- Rotated MNIST", due pannelli (incertezze vs angolo; accuratezza vs angolo) più una striscia di cifre di esempio lungo la sweep.*

La tabella è verificabile direttamente nell'output salvato della cella 12 del notebook come sta su disco (esecuzione sequenziale completa, `execution_count` 1–8), e i valori coincidono con quelli citati in `TODO.md` §3b.

---

## 6. Risultati su UCI HAR

### 6.1 Training MAP del modello sorgente

`notebooks/har_source_training.ipynb`. Split train/val **per soggetto** dentro il pool sorgente (16 soggetti / 4 soggetti — `1, 2, 24, 26`): 2 510 finestre di training, 687 di validazione (cella 7). Bilanciamento di classe verificato sul pool effettivamente scelto e non solo globalmente: 910/839/761, rapporto max/min 1.20 → nessuna pesatura di classe applicata (cella 11). AdamW, lr $10^{-3}$, weight decay 0.01, early stopping su val loss con patience 20: arresto all'epoca 111, miglior epoca 91 (cella 15).

Sul validation sorgente: **accuratezza 0.975, macro-recall 0.974** (cella 17). La macro-recall è riportata assieme all'accuratezza perché con classi solo approssimativamente bilanciate l'accuratezza grezza da sola può nascondere il collasso di una classe; qui le due coincidono, il che è la conferma che serve.

Per soggetto target (cella 19):

| soggetto | 7 | 8 | 11 | 15 | 19 | 20 | 21 | 22 | 27 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|
| accuratezza | 0.935 | 0.913 | 0.987 | 0.993 | 0.779 | **0.653** | 0.979 | 0.895 | **1.000** | 0.995 |
| macro-recall | 0.936 | 0.924 | 0.989 | 0.994 | 0.804 | 0.643 | 0.980 | 0.902 | 1.000 | 0.995 |

Accuratezza media 0.913 ± 0.108, **range 0.653–1.000**. Il range è il dato importante, più della media: non c'è saturazione (esiste almeno un soggetto duro, il 20) e non c'è collasso al livello del caso (il minimo 0.653 è ben sopra 1/3). Ma va detto subito che **6 dei 10 target sono già ≥ 0.94 senza alcun adattamento**: la popolazione target è dominata da soggetti su cui non c'è nulla da guadagnare. Questa asimmetria è la ragione strutturale per cui la media macro dell'ablazione (§6.5) è fuorviante.

*Figura: `notebooks/har_source_training.ipynb`, cella 17 — sezione "7. Training Curves", tre pannelli (cross-entropy, accuratezza, macro-recall) train vs val con l'epoca migliore evidenziata.*

### 6.2 Convergenza Monte Carlo: un criterio tautologico trovato, corretto, e non soddisfatto

`notebooks/har_mc_convergence.ipynb`. Prima di usare la predittiva MC serve fissare $M$. Il criterio inizialmente naturale — «confronta ogni $M$ con il valore al massimo $M$ testato» — è **tautologico**: il valore più grande della lista viene confrontato con sé stesso e passa per costruzione. La versione corretta (documentata nella cella markdown 14) usa un riferimento **indipendente** $M_{\text{REFERENCE}} = 5000$, non presente nella lista testata, e richiede tre condizioni congiunte: deviazione relativa < 1% **e** deviazione assoluta < 0.0005 dal riferimento, **e** stabilità su una finestra di 3 valori consecutivi. Il seed del generatore è fissato (123) per tutti gli $M$, così che le differenze vengano da $M$ e non dal caso.

Il modello e la posterior sono ricostruiti deterministicamente da §4 e verificati: $\tau_{\text{prior}}$ identico a 25.10 (differenza 0.00) e autovalore minimo della covarianza coerente con §4 (celle 10 e 11).

Risultato (cella 19):

| $M$ | 10 | 25 | 50 | 100 | 200 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|---|---|---|
| dev. rel. su source val | 23.93% | 18.20% | 14.31% | 1.18% | 1.62% | 0.23% | 1.39% | 0.12% |
| dev. rel. su target 21 | 0.76% | 7.08% | 2.56% | 2.55% | 1.38% | 1.03% | 0.30% | 1.03% |
| criterio puntuale | no | no | no | no | no | no | no | no |

**Nessun $M$ soddisfa il criterio rigoroso.** Le deviazioni relative oscillano intorno all'1% senza scendere sotto in modo stabile su tre valori consecutivi: il rumore MC domina sull'andamento di convergenza a tutti gli $M$ testati, perché il segnale epistemico su questo dataset è piccolo in assoluto (~$5.8\times10^{-3}$ nat) e stabilizzare una quantità di quell'ordine richiede molti campioni. Il notebook prende la strada onesta: stampa «NO M in M_VALUES satisfied windowed convergence criterion» e adotta $M_{\text{FIXED}} = 5000$ = il riferimento stesso, cioè il valore più alto disponibile come stima più affidabile, dichiarando che il criterio *non* è stato soddisfatto.

Questo è riportato qui come esempio di rigore, non come risultato positivo travestito. Un criterio che *non* si supera e viene dichiarato tale è più informativo di uno tarato a posteriori per essere superato. La conseguenza pratica è che tutte le quantità epistemiche di §6.3 sono calcolate con $M = 5000$ e che l'incertezza MC residua su di esse è dell'ordine dell'1% relativo — non trascurabile in assoluto, ma piccola rispetto alle differenze fra soggetti (che sono di ordini di grandezza, si veda §6.3).

*Figura: `notebooks/har_mc_convergence.ipynb`, cella 18 — sezione "4. Convergence Analysis and M Selection", griglia 2×3 (source val e target 21 × totale/epistemica/aleatoria) in funzione di $M$ in scala logaritmica, con la linea del riferimento $M = 5000$.*

### 6.3 Scomposizione dell'incertezza

`notebooks/har_uncertainty_decomposition.ipynb`, sull'artefatto di §6.2 ($M = 5000$, temperatura 1.0). Cella 7:

| soggetto | n | accuratezza | epistemica | aleatoria | totale |
|---|---|---|---|---|---|
| source val | 687 | 0.975 | 0.0058 | 0.0591 | 0.0649 |
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

Due osservazioni immediate. Primo: **l'aleatoria è più grande dell'epistemica dappertutto**, anche sul soggetto peggiore. Con 3 classi il tetto di entropia è $\log 3 = 1.10$ nat e le classi di locomozione hanno sovrapposizione reale già in-distribution; non c'è molto spazio per nessuno dei due termini. Secondo: è comunque **l'epistemica** il segnale che traccia la difficoltà.

**Check semantico A — soggetto più difficile → epistemico-dominante** (cella 10, con `assert`). Il soggetto con l'accuratezza più bassa (20, 0.653) è **anche** quello con l'epistemica più alta di tutta la coorte (0.0515), pari a **8.9×** il riferimento del validation sorgente (0.0058). Il check è formulato come *ranking*, non come dominanza assoluta dell'epistemica sull'aleatoria (che, per il punto precedente, non si verifica mai qui) — ed è la formulazione corretta della proprietà che si vuole verificare.

*Figura: `notebooks/har_uncertainty_decomposition.ipynb`, cella 11 — sezione "3. Semantic check -- hardest target subject", due pannelli: scatter epistemica vs accuratezza con etichette dei soggetti, e barre orizzontali di epistemica media ordinate (source val incluso).*

**Check semantico B — SITTING vs STANDING → aleatorio-dominante** (celle 13–15). Il modello a 3 classi non conosce le posture statiche, quindi questo check richiede un modello dedicato: stessa architettura ma `561 → 128 → 64 → 6`, stesso split per soggetto (verificato identico con `assert`), scaler rifittato, $\tau_{\text{prior}} = 54.42$, accuratezza su val sorgente 0.970. SITTING e STANDING sono l'esempio da manuale di incertezza *aleatoria* in HAR da accelerometro: due posture statiche con statistiche del segnale quasi identiche.

| gruppo | epistemica | aleatoria | rapporto ale/epi |
|---|---|---|---|
| SITTING + STANDING | 0.0126 | 0.1082 | 8.6× |
| classi ben separabili (WALKING, DOWNSTAIRS, LAYING) | 0.0050 | 0.0190 | 3.8× |
| **elevazione rispetto alle ben separabili** | **2.5×** | **5.7×** | — |

La confusione fra le due posture è dunque guidata dall'aleatoria (elevata 5.7×) e non dall'epistemica (2.5×): è ambiguità intrinseca delle feature, non mancanza di dati. La matrice di confusione conferma la localizzazione dell'errore (SITTING → STANDING 7 casi, STANDING → SITTING 8, SITTING → LAYING 13; le altre classi quasi diagonali, cella 14).

*Figura: `notebooks/har_uncertainty_decomposition.ipynb`, cella 16 — sezione "4. Semantic check -- SITTING vs STANDING", barre epistemica/aleatoria per ognuna delle 6 classi, con SITTING e STANDING evidenziate.*

**Normalizzazione su scala assoluta** (celle 18–19). I valori grezzi (da 0.0002 a 0.05 nat) non sono interpretabili da soli, e normalizzare *per batch* sarebbe attivamente ingannevole: un batch di soli soggetti difficili apparirebbe "normale" rispetto a sé stesso, cancellando esattamente il segnale cercato (`TODO.md` §1b: «never per-batch»). `normalize_epistemic` offre due riferimenti fissi: divisione per $\log K = 1.0986$, e z-score robusto contro la distribuzione epistemica del **validation sorgente** (mediana 0.000453, IQR 0.002205), calcolata una volta e riusata per ogni soggetto.

| soggetto | 27 | 30 | 15 | 21 | source val | 11 | 7 | 22 | 8 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| z-score | −0.12 | 0.94 | 1.02 | 1.89 | 2.43 | 2.49 | 7.36 | 10.12 | 17.40 | 20.06 | **23.17** |

Su una scala fissata una volta sola dai dati sorgente, i soggetti facili sono indistinguibili dal rumore del validation sorgente ($z \lesssim 2.5$) e quelli difficili stanno a $z > 17$: un rapporto di ordini di grandezza, non un effetto marginale.

*Figura: `notebooks/har_uncertainty_decomposition.ipynb`, cella 19 — sezione "5. Absolute normalization", barre orizzontali dei z-score ordinati con la linea dello zero.*

**Effetto di $\tau$** (celle 21–22). Qui $\tau$ è la `temperature` della predittiva (l'iperparametro fissato a 0.4 in `TODO.md` §1), non la precisione della prior. Sweep su source val e sul soggetto più difficile:

| $\tau$ | 0.1 | 0.2 | **0.4** | 0.7 | 1.0 | 1.5 | 2.0 | 4.0 |
|---|---|---|---|---|---|---|---|---|
| frazione epistemica, source val | 0.757 | 0.555 | **0.303** | 0.147 | 0.090 | 0.057 | 0.044 | 0.023 |
| frazione epistemica, soggetto 20 | 0.873 | 0.753 | **0.552** | 0.348 | 0.230 | 0.130 | 0.084 | 0.029 |

La frazione epistemica cala **monotonamente** da $\tau = 0.1$ a $\tau = 4$ per entrambi. Il meccanismo (cella markdown 23): dividere i logit per $\tau < 1$ rende ogni singolo campione della posterior più vicino a one-hot (entropia per campione più bassa → aleatoria giù) e amplifica il disaccordo *fra* campioni (epistemica su); oltre $\tau = 1$ entrambi gli effetti si invertono. Il valore fissato $\tau = 0.4$ tiene l'epistemica intorno a metà del segnale totale per il soggetto difficile (0.552), contro solo il 23% al default non toccato $\tau = 1$, senza l'instabilità del regime $\tau < 0.2$ dove la decomposizione diventa quasi tutta epistemica e smette di discriminare fra soggetti.

*Figura: `notebooks/har_uncertainty_decomposition.ipynb`, cella 22 — sezione "6. Effect of τ", due pannelli: frazione epistemica vs τ (scala log) con la linea a τ = 0.4, e magnitudini grezze su source val.*

### 6.4 Calibrazione: il risultato chiave

`notebooks/har_calibration.ipynb`. **Protocollo multi-seed** (`TODO.md` §1, «5 seeds minimum for any reported correlation/ranking»): lo split sorgente/target e quello train/val sono **fissi** (seed 42) — cambiare *chi* è sorgente e chi target non è la variabilità su cui si vuole mediare; ciò che varia sui 5 seed (0–4) è l'inizializzazione e il training del modello, che è ciò che effettivamente produce stime di calibrazione e di epistemica diverse. $M = 1000$ campioni MC, ECE a 15 bin, diagrammi di affidabilità a 8 bin (con area del marker proporzionale al conteggio del bin, così che un bin con pochi campioni si legga come rumore e non come segnale).

**ECE, NLL, Brier — media ± std su 5 seed** (celle 12–13):

| soggetto | accuratezza | ECE (MAP) | ECE (Laplace) | NLL (MAP) | NLL (Laplace) | Brier (MAP) | Brier (Laplace) |
|---|---|---|---|---|---|---|---|
| source val | 0.9758 ± 0.0027 | 0.0104 ± 0.0030 | 0.0113 ± 0.0024 | 0.0504 | 0.0516 | 0.0305 | 0.0305 |
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

Sui quattro rimanenti (source val, 15, 27, 30) la Laplace **peggiora leggermente** un ECE già minuscolo (per es. source val 0.0104 → 0.0113): con un modello già ben calibrato, la dispersione aggiuntiva della posterior sovracorregge. È un risultato onesto e coerente col meccanismo: la Laplace guadagna dove la stima puntuale è troppo confidente, e non ha niente da correggere dove già non lo è.

*Figura: `notebooks/har_calibration.ipynb`, cella 10 — sezione "4. Reliability diagrams, MAP vs. Laplace", griglia 3×4 con un pannello per source val e per ognuno dei 10 target (predizioni aggregate sui 5 seed, MAP in rosso e Laplace in blu, area del marker = conteggio del bin, proxy di shift nel titolo).*

**Il risultato chiave: ρ = 0.84 ± 0.03** (cella 21). Per ogni seed indipendentemente si correlano, sui 10 soggetti target, il proxy di shift e l'epistemica media; si media poi il *coefficiente* (non i valori epistemici, che aggregati prima cancellerebbero la variabilità che si vuole misurare — è ciò che `TODO.md` §1 chiede letteralmente):

| seed | 0 | 1 | 2 | 3 | 4 | media ± std |
|---|---|---|---|---|---|---|
| ρ di Spearman | 0.855 | 0.867 | 0.855 | 0.855 | 0.782 | **0.842 ± 0.031** |
| p | 0.0016 | 0.0012 | 0.0016 | 0.0016 | 0.0075 | tutti < 0.01 |

Il coefficiente è stabile (std 0.031 su cinque inizializzazioni indipendenti) e significativo su ogni seed preso singolarmente. Per confronto, la stessa correlazione con l'ECE della Laplace, calcolata sulle medie fra seed, vale ρ = 0.806 (p = 0.0049) (cella 24) — anche la calibrazione degrada con lo shift, ma l'epistemica è il segnale più netto dei due.

Un disaccordo merita di essere segnalato, non appianato: il soggetto **20 ha un proxy di shift *minore* del 19** (18.43 vs 29.21) ma **epistemica maggiore** (0.0515 vs 0.0447) e accuratezza minore (0.653 vs 0.779). Il proxy a centroide su 561 dimensioni non è la verità di riferimento; qui l'epistemica traccia la difficoltà *reale* meglio del proxy. Detto in modo rigoroso: ρ = 0.84 misura l'accordo con un proxy grezzo, e l'unico caso in cui i due si discostano è un caso in cui il proxy sbaglia. Questo rafforza il risultato, ma va enunciato come osservazione su un singolo soggetto, non come dimostrazione.

*Figure: `notebooks/har_calibration.ipynb`, cella 22 — sezione "8. Spearman(shift proxy, epistemic uncertainty)", scatter con barre d'errore sui 5 seed ed etichette dei soggetti. Cella 24 — sezione "9. Summary plots", due pannelli affiancati: ECE vs shift ed epistemica vs shift, stesso asse x.*

**La sfumatura sull'AUROC di rilevamento degli errori** (celle 16–17). Qui il quadro è più articolato e va riportato per intero:

| soggetto | AUROC entropia MAP | AUROC totale (Laplace) | AUROC epistemica |
|---|---|---|---|
| source val | **0.9880** | 0.9869 | 0.9773 |
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

L'epistemica **non** è uniformemente il miglior rilevatore di errore. Sui soggetti 7 e 8 — moderatamente spostati secondo il proxy ma ancora accurati al 92–95% — l'entropia MAP e la totale la superano leggermente, perché lì la maggior parte degli errori nasce da sovrapposizione reale fra classi, cioè è aleatoria. Sui due soggetti con l'accuratezza **più bassa in assoluto** — 20 (68%) e 19 (77%), i davvero difficili per esito — l'epistemica è invece la **migliore** delle tre (20: 0.811 > 0.770 > 0.733; 19: 0.893 > 0.857 > 0.802).

La formulazione corretta è quindi: **l'epistemica è prima un rilevatore di shift, poi di errore.** Traccia in modo affidabile la difficoltà del dominio (§6.3, e ρ = 0.84 qui sopra); se sia anche il miglior rilevatore di errore su un dato soggetto dipende da quale tipo di incertezza domini gli errori là. È una distinzione che l'entropia totale, mescolando i due termini, non permette di fare.

**Accuratezza vs copertura** (cella 19). Ordinando i punti per entropia totale crescente e calcolando l'accuratezza sulla frazione più confidente, la curva sale al ridursi della copertura anche sul soggetto a shift più alto: il soggetto 19 parte da 0.768 a copertura piena (valore dalla tabella multi-seed sopra) e la curva tende verso ≈1.0 a copertura bassa. **Il valore ≈1.0 è una lettura della curva, non un numero stampato** dal notebook (`TODO.md` §8 riporta la stessa lettura). Operativamente è il risultato più direttamente rilevante per uno scenario assistivo: se il sistema potesse astenersi quando è incerto, l'accuratezza sulle risposte che decide di dare tornerebbe vicina a quella sorgente.

*Figura: `notebooks/har_calibration.ipynb`, cella 19 — sezione "7. Accuracy-vs-coverage curve", quattro curve (source val e i soggetti a shift basso/medio/alto: 27, 11, 19), reiezione per entropia totale della Laplace.*

### 6.5 Ablazione dell'adattamento

`notebooks/har_adaptation_ablation.ipynb`. Sei bracci, **stessi seed, stessi dati, stesso ottimizzatore**: cambia solo l'obiettivo. 5 seed × 10 target × 6 bracci = 300 run.

| braccio | $\gamma$ | peso $w_i$ |
|---|---|---|
| (a) nessun adattamento | — | — (modello sorgente congelato) |
| (b) sola entropia | 0 | 1 (nessun termine di diversità — collasso atteso) |
| (c) IM senza pesatura (SHOT-IM) | 0.5 | 1 |
| (d) IM + peso da entropia predittiva MAP | 0.5 | $\exp(-H^{\text{MAP}}_i)$ |
| (e) IM + peso da entropia predittiva Laplace (fedele al paper) | 0.5 | $\exp(-H^{\text{totale}}_i)$ |
| (f) IM + peso da epistemica BALD standardizzata (**deviazione**) | 0.5 | $\exp(-z_i)$, $z_i$ contro riferimento sorgente fisso |

Il braccio (d) è il controllo cruciale: pesare *in qualche modo* i campioni potrebbe già bastare, e senza (d) non si potrebbe attribuire un eventuale vantaggio al trattamento bayesiano.

**Risultati macro-mediati sui 10 target, media ± std sui 5 seed** (cella 10):

| braccio | accuratezza | Δacc vs (a) | ECE | NLL | collasso >90% una classe | **classe azzerata (collasso parziale)** |
|---|---|---|---|---|---|---|
| (a) | 0.9157 ± 0.0022 | 0.0000 | 0.0713 | 0.3710 | 0.0% | 0.0% |
| (b) | 0.5143 ± 0.0287 | −0.4014 | 0.4857 | **13.3756** | **34.0%** | 80.0% |
| (c) | 0.8667 ± 0.0231 | −0.0490 | 0.1333 | 3.6017 | 0.0% | 14.0% |
| (d) | 0.8845 ± 0.0482 | −0.0312 | 0.1155 | 3.1154 | 0.0% | 22.0% |
| **(e)** | **0.8945 ± 0.0401** | **−0.0212** | **0.1055** | **2.8405** | 0.0% | **24.0%** |
| (f) | 0.8779 ± 0.0385 | −0.0378 | 0.1220 | 3.2541 | 0.0% | 30.0% |

Il controllo negativo (b) fa quello che deve: collassa completamente (>90% delle predizioni su una classe) nel 34% dei run, con NLL ~13.4. L'ordinamento dei bracci pesati è (e) > (d) > (f) > (c), coerente col paper: la pesatura bayesiana fedele al paper è la migliore, quella con l'incertezza del solo modello puntuale la segue, l'IM non pesato è il peggiore dei quattro.

Ma **macro-mediato su tutti i 10 target, ogni braccio di adattamento è peggiore del non fare niente**, incluso (e). Non è un bug: 6 dei 10 target sono già ≥94% (§6.1), quindi la media è dominata da «l'adattamento sposta un modello già corretto dal suo ottimo» e nasconde «l'adattamento salva un soggetto genuinamente spostato» come effetto di minoranza.

**Breakdown per tercile di shift** (cella 17). Terzili per proxy di shift: basso `27, 15, 21`, medio `7, 30, 11`, alto `22, 20, 8, 19`. I due terzili basso e medio sono esattamente i 6 soggetti già ≥94%; l'alto è i 4 genuinamente difficili (68–92%).

| braccio | Δacc tercile basso | Δacc tercile medio | Δacc tercile **alto** |
|---|---|---|---|
| (b) | −0.2374 ± 0.0522 | −0.5100 ± 0.1220 | −0.4429 ± 0.0279 |
| (c) | −0.0339 ± 0.0516 | −0.1237 ± 0.1061 | −0.0043 ± 0.0387 |
| (d) | −0.0178 ± 0.0233 | −0.1302 ± 0.0374 | **+0.0331 ± 0.1031** |
| **(e)** | −0.0164 ± 0.0225 | −0.1325 ± 0.0400 | **+0.0586 ± 0.1024** |
| (f) | −0.0043 ± 0.0048 | −0.0809 ± 0.0576 | −0.0306 ± 0.0656 |

**(d) ed (e) sono gli unici bracci che diventano positivi, e solo sul tercile alto** — esattamente i 4 soggetti che avevano bisogno di adattamento. È la validazione qualitativa della tesi del paper su dati reali: la vista macro-mediata da sola è fuorviante, la vista per tercile è quella che valida il metodo.

**Va però detto con la stessa chiarezza che l'effetto non è statisticamente distinguibile da zero.** Il +0.0586 di (e) sul tercile alto ha una deviazione standard fra seed di ±0.1024, cioè l'intervallo comprende ampiamente lo zero; lo stesso vale per il +0.0331 di (d) (±0.1031). Su cinque seed e quattro soggetti, quello che si può affermare è che il *segno* dell'effetto è quello previsto e che l'ordinamento fra bracci è consistente, non che l'effetto sia significativo. Il tercile medio, curiosamente, è il più penalizzato per ogni braccio (peggio del basso): i soggetti `7, 30, 11` stanno al 94–99.5% con confini di decisione presumibilmente più stretti di `27, 15, 21` (praticamente al 100%), quindi toccare l'estrattore costa loro di più — un'osservazione riportata nella cella markdown 18 del notebook.

Il braccio (f) resta negativo perfino sul tercile alto (−0.0306): standardizzare la sola epistemica, senza la componente aleatoria dell'entropia totale, non basta a identificare in modo affidabile dove l'adattamento sia da fidarsi. La deviazione dal paper sottoperforma il braccio fedele proprio dove conta. Nelle traiettorie i pesi di (f) risultano anche visibilmente più rumorosi.

**Il caveat del collasso parziale, riportato con lo stesso peso con cui è stato scoperto.** La colonna «collasso >90% una classe» legge 0% per tutti i bracci tranne (b), e sarebbe facile fermarsi lì. Ma quella soglia cattura solo il collasso *totale*. Una metrica distinta — «qualche classe riceve zero predizioni» — mostra collasso **parziale** nel **14% dei run di (c), 22% di (d), 24% di (e), 30% di (f)**. Il caso documentato: sul soggetto 19, il braccio (f) assegna zero predizioni a WALKING_DOWNSTAIRS in tutti i 5 seed. Il punto che conta è che **non è specifico di (f)**: la minimizzazione di entropia con testa congelata può azzerare una classe su *qualsiasi* braccio con $\gamma > 0$, con frequenza che cresce passando da (c) verso (f). Il 24% di (e) è un caveat reale sui suoi numeri aggregati, che sono i migliori della tabella: il braccio "vincente" azzera una classe in circa un run su quattro. Questa metrica è stata aggiunta in revisione, dopo che la prima lettura della tabella (che guardava solo la soglia >90%) suggeriva erroneamente che soltanto (b) collassasse.

*Figure: `notebooks/har_adaptation_ablation.ipynb`, cella 17 — sezione "7. Per-shift-tercile breakdown", barre di Δaccuratezza per tercile e braccio con barre d'errore sui 5 seed. Cella 15 — sezione "6. Per-arm trajectories", quattro pannelli (loss IM, termine di entropia pesato, termine di diversità, peso medio per campione) sui 300 passi per il soggetto a shift più alto, banda media ± std sui 5 seed. Cella 13 — sezione "5. Reliability diagrams by arm", sei pannelli, predizioni aggregate su 5 seed × 10 target.*

**Nota sulla riproducibilità di questa sezione.** Tutti i numeri sopra provengono dagli output salvati del notebook come sta su disco (celle 10 e 17), e coincidono con quelli citati in `TODO.md` §9. Va però segnalato che il modello sorgente **non è bit-identico fra esecuzioni successive** dello stesso notebook con gli stessi seed: l'early stopping può cadere a un'epoca diversa per effetto di non-determinismo in virgola mobile, e siccome ogni braccio parte da quel modello, i tassi di collasso e i Δ per tercile ereditano questa variabilità *oltre* a quella fra seed già riportata. Questa variabilità **fra run** non è quantificata in questo progetto: il protocollo a 5 seed × 10 soggetti stima la variabilità dovuta all'inizializzazione, non quella dovuta a ri-esecuzione. È una ragione ulteriore per leggere il +0.0586 del tercile alto come indicazione di segno e non come stima puntuale.

---

## 7. Estensione open-set (HAPT)

### 7.1 Che cosa si è cercato di testare

Il paper rivendica vantaggi anche nel setting **open-set**, dove il target contiene classi che il sorgente non ha mai visto (*target-private* o OOD): l'incertezza dovrebbe essere il meccanismo naturale per riconoscerle senza aver mai visto un esempio negativo in addestramento. Il progetto costruisce questo setting su **HAPT** (*Human Activities and Postural Transitions*), estensione naturale di UCI HAR: stessi 30 soggetti, stesse 561 feature, più 6 classi di **transizione posturale** (STAND_TO_SIT, SIT_TO_STAND, SIT_TO_LIE, LIE_TO_SIT, STAND_TO_LIE, LIE_TO_STAND).

Verifica del dataset (`har_openset.ipynb`, celle 4–5): 10 929 finestre, 561 feature, 30 soggetti (ID 1–30, coincidenti con UCI HAR, verificato con `assert`), 0 NaN/Inf, valori in $[-1,1]$; conteggi delle transizioni 70/33/107/85/139/84.

Il setup: in-distribution = attività di locomozione dei 10 soggetti target da UCI HAR (1 475 finestre); OOD = transizioni posturali **degli stessi soggetti** da HAPT (160 finestre, da 11 a 25 per soggetto). Modello: il classificatore a 3 classi di §6.1, ricostruito su 5 seed di training (42, 123, 456, 789, 1011), $M = 5000$. Nessun riaddestramento.

### 7.2 Il risultato sorprendente

AUROC con in-distribution = 1, OOD = 0 e score = incertezza negata; un rilevatore che funziona dovrebbe stare sopra 0.5 (cella 10):

| score | AUROC (media ± std su 5 seed) | rapporto OOD / in-dist |
|---|---|---|
| epistemica | **0.0938 ± 0.0246** | 0.325 ± 0.061 |
| aleatoria | 0.0888 ± 0.0196 | 0.170 ± 0.010 |
| totale | 0.0899 ± 0.0215 | 0.205 ± 0.007 |

L'AUROC non è basso: è **invertito**, e stabilmente (std ≤ 0.025 su 5 seed). Il modello è **più certo** sulle transizioni posturali che sulle attività di locomozione che dovrebbe conoscere: l'incertezza epistemica sulle OOD è appena il 32.5% di quella sulle in-distribution. Un valore di 0.09 su 5 seed non è rumore né un errore di segno nel calcolo (lo score è coerentemente negato per tutte e tre le quantità, e tutte e tre concordano).

### 7.3 La diagnosi: due ipotesi e un controllo controfattuale

Il notebook formula due ipotesi alternative (celle 8 e 10):

1. **Limite di interpolazione della Laplace.** Le transizioni posturali, nello spazio delle 561 feature standardizzate sul sorgente, cadono **dentro** l'inviluppo convesso dei dati sorgente — non fuori. Sono movimenti brevi e a bassa energia, statisticamente più vicini al centro della distribuzione di quanto lo siano le camminate di un soggetto con andatura atipica. Una Laplace sull'ultimo strato rileva bene l'**estrapolazione** (punti lontani dal supporto, dove campioni diversi della posterior divergono) e per costruzione **non** rileva la novità semantica **interpolata**: se una classe nuova cade in una regione ben coperta dai dati sorgente, la posterior sulla testa è là stretta e concorde, e il modello è confidente. Con feature congelate a $\beta_{\text{MAP}}$, nulla nel meccanismo può segnalare "questa è una classe che non ho mai visto" se le sue feature assomigliano a qualcosa che ha visto.
2. **Riferimento confondente.** Il riferimento in-distribution usato sono i soggetti *target*, che sono a loro volta spostati. Il segnale di novità di classe potrebbe essere semplicemente mascherato dallo shift inter-soggetto del riferimento, che alza l'incertezza sulle in-distribution.

Le due ipotesi fanno previsioni **diverse e distinguibili**, e il notebook esegue il controllo che le separa: rifare lo stesso AUROC con il **validation sorgente** come riferimento in-distribution (stesso modello, stessa posterior, stesso gruppo OOD, solo il riferimento cambia). Se valesse l'ipotesi 2, l'AUROC dovrebbe risalire sopra 0.5 rimuovendo il confondente; se vale l'ipotesi 1, dovrebbe restare invertito.

| riferimento in-distribution | AUROC epistemica | rapporto epistemico OOD / in-dist |
|---|---|---|
| attività note dei soggetti **target** | 0.0938 ± 0.0246 | 0.325 ± 0.061 |
| **validation sorgente** | **0.0799 ± 0.0171** | **1.062 ± 0.142** |

**L'ipotesi 1 è confermata, la 2 esclusa.** Con il riferimento pulito l'AUROC resta invertito (0.0799), quindi il confondente non spiegava il risultato. E il rapporto racconta la storia in modo ancora più diretto: l'epistemica media sulle transizioni è **1.06×** quella del validation sorgente, cioè **statisticamente indistinguibile da dati che il modello ha effettivamente visto in addestramento** — mentre le stesse attività di locomozione dei soggetti target arrivano fino a 8.9× (§6.3). Il modello tratta una classe che non ha mai visto come se fosse dominio noto. La logica di questo confronto è codificata nel notebook stesso (cella 10), che stampa «Both references show inverted AUROC (<0.5) → This supports Hypothesis (1): Laplace interpolation limit. HAPT transitions fall within feature space convex hull.»

### 7.4 Perché è un contributo e non solo un risultato negativo

Il valore di questa sezione non è l'AUROC di 0.09 in sé: è che il progetto ha **isolato e diagnosticato un limite reale del metodo con un controllo pulito**. La differenza fra "l'esperimento open-set non ha funzionato" e quanto sopra è la struttura del ragionamento: due ipotesi concorrenti con previsioni distinguibili, un controllo controfattuale che cambia una sola variabile, una conclusione che ne segue e una che viene esclusa, il tutto stabile su 5 seed.

La conseguenza sostanziale per il metodo è precisa e non era ovvia a priori: **l'incertezza epistemica da Laplace sull'ultimo strato è un rilevatore di estrapolazione, non di novità semantica.** Sui benchmark del paper (Office-Home, VisDA-C) le classi target-private sono categorie visive diverse, che una ResNet-50 mappa presumibilmente fuori dal supporto sorgente, e il meccanismo funziona. Con feature pre-ingegnerizzate e classi nuove che sono *interpolazioni* di quelle note, non funziona — e non può, per come è costruito. È un limite del tipo di rappresentazione, non dell'implementazione.

**Due lacune di questa sezione, da dichiarare.** `TODO.md` §10 marca come completati due item che **non sono presenti nel notebook eseguito**: (i) il report dell'accuratezza OS (inclusa la classe "unknown") e OS\* (sole classi condivise), che sono le metriche standard del setting open-set usate dal paper (§4 di U-SFAN, protocollo di SHOT); (ii) il fallback che rimuove LAYING dal sorgente a 6 classi mantenendolo nel target. Il notebook realizza uno studio di **rilevamento OOD via AUROC**, che è cosa diversa da un'accuratezza open-set con soglia. Inoltre le dimensioni campionarie OOD sono piccole (160 finestre totali, 11–25 per soggetto), ai limiti della soglia di 10 che il notebook stesso si dà.

`[DA DISCUTERE INSIEME: se aggiungere OS/OS* con una soglia sull'incertezza (breve da fare, riusando l'artefatto già salvato in data/har_bald_artifact_openset.npz), o se ridichiarare l'ambito di §10 come "rilevamento OOD" e correggere di conseguenza le spunte in TODO.md. Data la diagnosi di §7.3, un'accuratezza OS basata su soglia sull'incertezza sarebbe prevedibilmente vicina al caso — che è comunque un risultato riportabile e coerente.]`

---

## 8. Limitazioni

### 8.1 Staleness della posterior durante l'adattamento

È il limite concettualmente più profondo, ed è **strutturale al metodo del paper**, non un difetto di questa implementazione. La posterior $q(\theta) = \mathcal{N}(\theta_{\text{MAP}}, H^{-1})$ è calcolata una volta sola, con l'estrattore congelato a $\beta_{\text{MAP}}$ e sommando su feature $\phi_n = g_{\beta_{\text{MAP}}}(x_n)$ dei dati sorgente. Durante l'adattamento $\beta$ **si muove** (è l'unica cosa che si muove), quindi le feature su cui la posterior era stata calibrata non esistono più. Formalmente: $H$ dipende da $\beta$ attraverso $\phi_n\phi_n^\top$, e dopo $k$ passi di adattamento la posterior corretta sarebbe quella di $\beta^{(k)}$, non quella di $\beta_{\text{MAP}}$.

L'implementazione mitiga in parte il problema — `laplace.predictive(model, X_target, ...)` è richiamata a **ogni passo** e valuta le feature *correnti*, quindi i pesi non sono congelati al passo 0 — ma $\theta_{\text{MAP}}$ e $H^{-1}$ restano quelli di $\beta_{\text{MAP}}$. Si campionano teste plausibili per un estrattore che non c'è più. Dopo 300 passi con lr $10^{-2}$ la deriva può essere sostanziale, e l'entità della staleness **non è misurata** in questo progetto. Un'estensione naturale sarebbe rifittare la Laplace periodicamente durante l'adattamento — ma sarebbe una deviazione dal paper, che fissa esplicitamente la posterior (Fig. 3b: «we keep the posterior over the parameters fixed»), e richiederebbe i dati sorgente, che nel setting source-free non ci sono. È quindi un limite intrinseco al setting, che vale la pena enunciare come tale.

### 8.2 Natura locale e unimodale della Laplace, mostrata empiricamente

Questo limite è di solito solo citato; qui è **visibile nei dati**, ed è il motivo per cui il test MCMC di §5.2 è più informativo di un semplice "✓ passato".

Il problema di test è **quasi separabile**: 60 punti in tre blob gaussiani ben distanziati, con accuratezza MAP 0.983 e test 1.000. In questo regime la log-likelihood è quasi piatta lungo le direzioni che aumentano il margine — spingere i pesi più in là non peggiora l'adattamento ai dati, e solo la prior li trattiene. La posterior vera è quindi **asimmetrica** lungo quelle direzioni, con moda diversa dalla media, mentre la Laplace è per costruzione una gaussiana **simmetrica centrata sulla moda**.

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

La conseguenza a valle è il bias sistematico di §5.2: tutte le incertezze predittive della Laplace stanno **sopra** la diagonale rispetto a MCMC, con offset medio di 0.29–0.32 nat su totale e aleatoria. Da qui una conclusione operativa importante per come leggere §6: **l'ordinamento fornito dall'epistemica è validato, il suo livello assoluto no.** Il progetto, per fortuna, usa l'epistemica solo come ranking (Spearman, z-score su quantili sorgente fissi) e mai come valore assoluto interpretato — la validazione copre l'uso che se ne fa. Ma questo va detto esplicitamente, non lasciato implicito.

**Un caveat sul caveat, per onestà:** il riferimento MCMC è una catena Metropolis–Hastings a random walk di 5 000 campioni (dopo 1 000 di burn-in) in 12 dimensioni, con tasso di accettazione 0.299. Non sono stati calcolati diagnostici di convergenza ($\hat{R}$, ESS, autocorrelazioni) né sono state confrontate catene multiple da inizializzazioni diverse. Una catena mal mescolata sotto-stima la dispersione della posterior e potrebbe produrre marginali apparentemente asimmetrici: parte della discrepanza osservata potrebbe quindi essere dovuta al riferimento e non alla Laplace. L'asimmetria osservata è **coerente** con la quasi-separabilità, ma un test più rigoroso richiederebbe HMC/NUTS con diagnostici.

`[DA DISCUTERE INSIEME: quanto forte tenere questa affermazione. Le tre opzioni sono (i) tenerla come sopra, con il caveat esplicito sulla catena; (ii) rafforzarla aggiungendo diagnostici di convergenza al notebook di test — poche righe di codice, ma è un rerun; (iii) indebolirla a "compatibile con", rinunciando al claim "dimostrato empiricamente". Io propendo per (ii) se c'è tempo, altrimenti (i).]`

### 8.3 Il pivot BASAN → UCI HAR

Il progetto era stato concepito su un dataset sEMG di arto inferiore (BASAN), con un'ipotesi clinicamente motivata: addestrare su soggetti sani e adattare a soggetti con patologia diagnosticata al ginocchio, dove la biomeccanica alterata costituisce uno shift di dominio genuino e non arbitrario (`docs/project_overview_it.md`). Il dataset è stato accantonato per problemi di qualità del segnale grezzo, il cui recupero avrebbe richiesto un lavoro di elaborazione del segnale (segmentazione, filtraggio, rimozione di artefatti, estrazione di feature) sproporzionato rispetto al focus **metodologico** del corso — che riguarda l'inferenza bayesiana, non la pulizia di segnali biomedici.

Ciò che si è perso in questa scelta va detto chiaramente, perché non è marginale:

- **Lo shift è cambiato di natura.** Sano vs patologico è uno shift con un meccanismo causale noto e una severità potenzialmente graduabile clinicamente; la variabilità inter-soggetto su HAR è più mite (rapporto 1.20× rispetto al riferimento intra-sorgente, §4.3) e priva di una variabile di severità indipendente. Il proxy a distanza fra centroidi è un surrogato grezzo, come mostra il disaccordo sul soggetto 20 (§6.4).
- **Il regime sperimentale è diventato quello sfavorevole per il metodo.** Il paper mostra i suoi vantaggi maggiori sotto shift *forte*; su HAR 6 target su 10 sono già ≥94% e non c'è nulla da adattare. Questa è la ragione strutturale per cui l'effetto di §6.5 è piccolo e non significativo — non un difetto di implementazione, ma una conseguenza diretta della scelta di dataset.
- **La motivazione applicativa si è indebolita.** L'argomento assistivo (un dispositivo indossabile che non deve essere sicuro quando sbaglia) resta valido come inquadramento, ma HAR non lo mette alla prova.

Il pivot è stata la scelta giusta rispetto agli obiettivi del corso — un pipeline bayesiano completo, validato e onestamente riportato su un dataset pulito vale più di un pipeline incompleto su un dataset problematico — ma va contabilizzato come un ridimensionamento della domanda di ricerca, non come una sostituzione neutra.

### 8.4 Il limite di interpolazione (§7)

L'incertezza epistemica da Laplace sull'ultimo strato **non rileva classi nuove che cadono dentro l'inviluppo delle feature sorgente**: AUROC 0.08–0.09 su 5 seed, con l'epistemica sulle OOD pari a 1.06× quella del validation sorgente. Diagnosticato con controllo controfattuale (§7.3). Limita il metodo al rilevamento di **estrapolazione**, che è ciò che serve nel setting closed-set con shift covariato, ma non nell'open-set con classi semanticamente nuove ma statisticamente interne. Su feature pre-ingegnerizzate come le 561 di HAR/HAPT questo limite è particolarmente stretto, perché la rappresentazione non è appresa per separare classi che non erano nel task sorgente.

### 8.5 Il collasso parziale nel 14–30% dei run (§6.5)

La metrica di collasso ovvia (>90% delle predizioni su una classe) legge 0% per tutti i bracci pesati e nasconde il problema. La metrica più stretta (qualche classe con zero predizioni) scatta sul 14% dei run di (c), 22% di (d), **24% di (e)** e 30% di (f). Il braccio fedele al paper, quello con i migliori numeri aggregati, azzera una classe in circa un run su quattro. La minimizzazione di entropia con testa congelata può azzerare una classe su qualsiasi braccio con $\gamma > 0$; il termine di diversità $\mathcal{L}_{\text{div}}$ a $\gamma = 0.5$ evita il collasso *totale* ma non quello parziale. In un contesto assistivo, un classificatore di modalità locomotoria che smette del tutto di prevedere "discesa di scale" sarebbe un fallimento operativo, non una degradazione graduale — quindi questo caveat pesa più di quanto suggerisca la sua entità sull'accuratezza aggregata.

A questo si aggiunge il fatto che questi tassi sono stimati su 50 run per braccio (5 seed × 10 soggetti) e con un modello sorgente che non è bit-identico fra ri-esecuzioni del notebook (§6.5): la loro incertezza è quindi maggiore di quanto suggerisca una singola cifra percentuale. Il protocollo non è abbastanza potente per stimarli con precisione, e la conclusione robusta è ordinale — il collasso parziale è presente su tutti i bracci con $\gamma > 0$ e cresce da (c) verso (f) — non la singola percentuale.

### 8.6 Popolazione e sensore di UCI HAR

I 30 soggetti sono **giovani adulti sani** (19–48 anni), con smartphone montato **alla vita** in condizioni di laboratorio. Le implicazioni:

- **Nessuna patologia, nessuna alterazione dell'andatura**: la variabilità inter-soggetto qui è variazione normale, non biomeccanica alterata. Il ridimensionamento rispetto all'ipotesi originale è quello discusso in §8.3.
- **Non è un esoscheletro.** Il nome del progetto (*BayesianExoAdaptation*) descrive l'inquadramento motivazionale, non il setup sperimentale: un dispositivo di assistenza al movimento avrebbe sensori multipli su segmenti corporei, dinamica di accoppiamento con l'utente, e vincoli di latenza. Qui la classificazione è **offline**, su finestre di 2.56 s già estratte.
- **Feature pre-ingegnerizzate e non riapprese.** Le 561 feature sono fisse; non c'è apprendimento di rappresentazione dai segnali inerziali grezzi (che il dataset contiene, e che `TODO.md` §12 elencava come possibile estensione). Come mostra §7, la rigidità della rappresentazione è direttamente responsabile del limite di interpolazione.
- **Nessuna asimmetria di costo.** Tutti gli errori pesano uguale nelle metriche riportate, mentre in un contesto assistivo confondere "discesa di scale" con "camminata in piano" ha conseguenze diverse dal confonderle nell'ordine opposto. La curva accuratezza-copertura di §6.4 è il passo più vicino a un trattamento asimmetrico (astensione invece di risposta), ma non è una vera analisi di costi.

### 8.7 §6 (KFAC) non implementato

Deliberato e motivato: con una testa di 195 parametri l'Hessiana esatta si inverte direttamente e KFAC introdurrebbe un'ipotesi di indipendenza (la somma $\sum_n \Lambda_n \otimes \phi_n\phi_n^\top$ non fattorizza) per ottenere un'approssimazione peggiore di una soluzione esatta già disponibile, senza alcun vantaggio computazionale dimostrabile a questa scala (§4.1). La conseguenza è che il progetto **non** ha una misura dell'errore introdotto da KFAC (varianza predittiva, ECE, correlazione di ranking BALD) rispetto all'esatta, che era l'ultimo item di `TODO.md` §6 e che sarebbe stato di per sé interessante. Se un revisore volesse quel confronto, richiederebbe una rete con testa molto più larga per essere significativo.

### 8.8 Altre limitazioni minori ma dichiarate

- **Criterio di convergenza MC non soddisfatto** (§6.2): $M = 5000$ è scelto come miglior valore disponibile, non come valore convergente secondo il criterio dichiarato. Il rumore MC residuo sulle stime epistemiche è dell'ordine dell'1% relativo.
- **Convenzione su $\tau_{\text{prior}}$**: implementata come $\text{wd}\times N_{\text{train}} = 25.10$ anziché $\text{wd}\times N_{\text{source}} = 31.97$, mentre l'Hessiana somma su tutte le 3 197 finestre sorgente (§4.5).
- **Due temperature non coincidenti nell'adattamento**: i logit della loss usano $\tau = 0.4$, la predittiva che genera i pesi usa il default 1.0 (§4.4). Non è chiaro se sia intenzionale.
- **Split sorgente/target singolo**: i 5 seed variano solo l'inizializzazione del modello, non la partizione dei soggetti. La LOSO completa sui 30 soggetti, prevista come opzione in `TODO.md` §1, non è stata eseguita, quindi tutti i risultati sono condizionati a una particolare assegnazione 20/10.
- **Un solo soggetto per il tracciamento delle traiettorie** (il 19, a shift massimo): le traiettorie di §6.5 non sono rappresentative dell'intera coorte.
- **Percorso assoluto locale** in un output salvato di `har_openset.ipynb` (cella 13), lasciato deliberatamente (`REFACTOR_NOTES.md`, finding 5).

---

## 9. Conclusione

La catena causale del progetto, end-to-end, con i numeri chiave e senza sovrastimarli.

**Lo strumento è corretto.** L'Hessiana esatta a $K$ classi si riduce alla formula binaria delle dispense §7.4 entro $5.7\times10^{-6}$ in assoluto, con la struttura a blocchi attesa (§5.1). La predittiva concorda con Metropolis–Hastings sulla posterior vera di un problema a 12 parametri con correlazioni di Pearson 0.94 (totale), 0.95 (aleatoria), 0.72 (epistemica) — con un bias sistematico di livello di ~0.3 nat che rende validato l'**ordinamento** ma non la scala assoluta (§5.2, §8.2). La covarianza è definita positiva ed esattamente simmetrica su tutti i problemi testati.

**Il meccanismo funziona dove lo shift è forte e inequivocabile.** Sul toy 2D, sotto shift forte, l'IM convenzionale non recupera nulla (0.682 → 0.684; sulla classe spostata 6.0% → 7.3%) mentre l'IM guidato dall'incertezza porta la classe spostata da 6.0% a 98.7% — con il controllo negativo `mild` che resta a 1.000 (§5.3, in un regime di iperparametri scelto perché il fallimento fosse visibile, e dichiarato come tale). Su MNIST vs Fashion-MNIST è specificamente la componente **epistemica** a portare il segnale: rapporto 10.9×, AUROC 0.987 contro 0.946 dell'entropia MAP grezza e 0.979 dell'entropia totale (§5.4). La decomposizione non è un ornamento: separa un segnale utile da uno che lo diluisce.

**Su uno shift naturale mite l'incertezza sa di cosa parla, e produce un effetto reale ma statisticamente più piccolo.** Su UCI HAR l'epistemica traccia il proxy di shift con **ρ = 0.84 ± 0.03 su 5 seed, tutti p < 0.01** (§6.4); il soggetto più difficile della coorte è anche quello con l'epistemica più alta, a 8.9× il riferimento sorgente, e su una scala assoluta fissata dai quantili sorgente i difficili stanno a $z > 17$ contro $z \lesssim 2.5$ dei facili (§6.3). I check semantici passano nella direzione prescritta dalla teoria (lontano dai dati → epistemico; SITTING/STANDING → aleatorio, elevazione 5.7× contro 2.5×). La Laplace migliora ECE e NLL su 7 soggetti su 11, con i guadagni concentrati sui difficili (soggetto 20: NLL 1.55 → 1.03) e una lieve sovracorrezione sui già ben calibrati (§6.4). Sull'adattamento, i bracci con pesatura da incertezza sono gli **unici** che diventano positivi, e **solo** sul tercile di shift alto (+0.06 per il braccio fedele al paper, +0.03 per quello con incertezza MAP) — ma con deviazione standard fra seed (±0.10) che comprende lo zero, e con un collasso parziale di classe nel 24% dei run del braccio migliore (§6.5). L'affermazione sostenibile è: **il segno e l'ordinamento dell'effetto sono quelli previsti dal paper, la sua significatività statistica su questi dati no.**

**E c'è un limite riconosciuto e diagnosticato.** L'estensione open-set su HAPT produce un AUROC invertito (0.09, stabile su 5 seed): l'incertezza epistemica non vede affatto le transizioni posturali come classi nuove. Il controllo controfattuale con il validation sorgente come riferimento esclude l'ipotesi del riferimento confondente (AUROC resta 0.08) e conferma quella del **limite di interpolazione**: l'epistemica sulle OOD è 1.06× quella dei dati di addestramento, cioè indistinguibile da dominio noto (§7). La Laplace sull'ultimo strato è un rilevatore di **estrapolazione**, non di novità semantica interpolata. Questo non è un fallimento dell'esperimento, è la sua conclusione: un limite reale del metodo, isolato con un controllo che cambia una sola variabile.

Il filo che tiene insieme i quattro blocchi è quello annunciato nel piano originale del progetto, e ogni anello regge o cede in modo documentato: c'è davvero uno shift da studiare (sì, ma mite in media e molto eterogeneo, §4.3); il punto di partenza è affidabile (sì, 0.975 su validation sorgente, con range 0.653–1.000 sui target, §6.1); l'incertezza del modello sa di cosa parla (sì, ρ = 0.84, con validazione MCMC dell'ordinamento e non del livello, §6.4); quella conoscenza si traduce in un adattamento migliore quando conta (**direzionalmente sì, ma non in modo statisticamente distinguibile su questi dati, e con un caveat di collasso parziale nel 24% dei run**, §6.5). E dove il meccanismo non può funzionare, il progetto lo ha mostrato e spiegato invece di ometterlo (§7).

---

## 10. Riferimenti

1. **Roy, S., Trapp, M., Pilzer, A., Kannala, J., Sebe, N., Ricci, E., Solin, A.** — *Uncertainty-guided Source-free Domain Adaptation*. European Conference on Computer Vision (ECCV), 2022. arXiv:2208.07591. — Paper replicato; copia locale in `paper/U-SFAN.pdf`. Riferimenti puntuali usati: §3 (definizione del problema, $f = h\circ g$, testa congelata in adattamento), Eq. 1–3 (SHOT-IM: cross-entropy con label smoothing, $\mathcal{L}_{\text{ent}}$, $\mathcal{L}_{\text{div}}$), Eq. 4–5 (posterior predittiva, Laplace sull'ultimo strato), Eq. 6 (integrazione MC), Eq. 7 ($\mathcal{L}^{\text{ug}}_{\text{ent}}$ con $w_i = \exp(-H)$), Fig. 2 (natura mode-seeking della Laplace; rilevamento OOD), Fig. 3b (posterior fissata durante l'adattamento), Fig. 4 (toy 2D, shift mite vs forte, confine di decisione ribaltato), §4 (protocollo open-set, metriche OS/OS\*).

2. **Kristiadi, A., Hein, M., Hennig, P.** — *Being Bayesian, Even Just a Bit, Fixes Overconfidence in ReLU Networks*. International Conference on Machine Learning (ICML), 2020, pp. 5436–5446. — Riferimento [26] del paper U-SFAN; base della Laplace sull'ultimo strato e dell'argomento sull'overconfidence delle reti ReLU. Replicato in scala ridotta in `notebooks/mnist_check.ipynb` (§5.4).

3. **Liang, J., Hu, D., Feng, J.** — *Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation* (SHOT). ICML, 2020. — Riferimento [35] del paper U-SFAN; origine dell'obiettivo IM usato come base (braccio (c) dell'ablazione) e del protocollo di valutazione open-set.

4. **Dispense del corso** — *Probabilistic Machine Learning*, Università degli Studi di Trieste (`PML_notes_full.pdf`, non versionato nel repository). Sezioni usate: §2.3 (entropia, divergenza KL, informazione mutua), §7.3 (approssimazione di Laplace), §7.4 (regressione logistica bayesiana, formula $S_N^{-1} = S_0^{-1} + \sum_n s_n(1-s_n)\phi_n\phi_n^\top$), §10.6 (Bayesian model averaging), cap. 8 (metodi Monte Carlo a catena di Markov).

5. **Anguita, D., Ghio, A., Oneto, L., Parra, X., Reyes-Ortiz, J. L.** — *A Public Domain Dataset for Human Activity Recognition Using Smartphones*. ESANN, 2013. — **UCI HAR**, UCI Machine Learning Repository n. 240: 30 soggetti, smartphone alla vita, 561 feature per finestra di 2.56 s, 10 299 finestre. Dati non versionati nel repository (si veda `README.md`).

6. **Reyes-Ortiz, J. L., Oneto, L., Samà, A., Parra, X., Anguita, D.** — *Transition-Aware Human Activity Recognition Using Smartphones*. Neurocomputing, 2016. — **HAPT** (*Smartphone-Based Recognition of Human Activities and Postural Transitions*), UCI Machine Learning Repository n. 341: estensione di UCI HAR con 6 classi di transizione posturale, usate come classi target-private nell'esperimento open-set (§7).

---

## Appendice A — Indice dei notebook e delle sezioni di `TODO.md`

| notebook | sezione `TODO.md` | contenuto | sezione di questo report |
|---|---|---|---|
| `loader.ipynb` | §2 | caricamento, validazione, split per soggetto, proxy di shift, PCA | §4.2–4.3 |
| `test_hessian_binary.ipynb` | §1b.1, §5 | validazione della riduzione $K=2$ | §5.1 |
| `test_laplace_vs_mcmc.ipynb` | §1b.2, §5 | validazione contro Metropolis–Hastings | §5.2, §8.2 |
| `synthetic_track.ipynb` | §3 | replica Fig. 4, check semantico, sweep di shift | §5.3 |
| `mnist_check.ipynb` | §3b | MNIST vs Fashion-MNIST, Rotated-MNIST | §5.4 |
| `har_source_training.ipynb` | §4 | training MAP, valutazione per soggetto, fit della Laplace | §6.1 |
| `har_mc_convergence.ipynb` | §5 | convergenza MC, generazione dell'artefatto BALD | §6.2 |
| `har_uncertainty_decomposition.ipynb` | §7 | decomposizione BALD, check semantici, normalizzazione, sweep di $\tau$ | §6.3 |
| `har_calibration.ipynb` | §8 | affidabilità, ECE/NLL/Brier, AUROC, copertura, Spearman | §6.4 |
| `har_adaptation_ablation.ipynb` | §9 | 6 bracci, terzili di shift, traiettorie | §6.5 |
| `har_openset.ipynb` | §10 | rilevamento OOD su HAPT, controllo controfattuale | §7 |
| — | §6 (KFAC) | **deliberatamente non implementato** | §4.1, §8.7 |

I due notebook di validazione dell'Hessiana si trovano in `notebooks/`, non in una directory `tests/` separata (che nel repository non esiste).

## Appendice B — Punti aperti raccolti

Riepilogo dei `[DA DISCUTERE INSIEME]` sparsi nel testo, in ordine di impatto sul report:

1. **§8.2** — quanto tenere forte l'affermazione sulla non-gaussianità della posterior, dato che la catena MH non ha diagnostici di convergenza.
2. **§7.4** — se aggiungere le metriche OS/OS\* e il fallback LAYING (marcati fatti in `TODO.md` §10 ma assenti dal notebook) o ridichiarare l'ambito di §10.
3. **§5.3** — se la prima versione non monotona della sweep può essere raccontata come risultato osservato, dato che non è conservata in git.
4. **§4.5** — se rieseguire con $\tau_{\text{prior}} = 0.01 \times 3197$ come controllo, o dichiarare la convenzione implementata.
5. **§6.4** — gli estremi esatti della curva accuratezza-copertura per il soggetto 19 sono letti dalla figura, non stampati; se serve un numero, va aggiunta una `print`.
6. **§6.5 / §8.5** — se quantificare la variabilità fra ri-esecuzioni (rieseguire l'ablazione *n* volte e riportare la dispersione dei tassi di collasso), oppure lasciarla dichiarata ma non misurata come sta ora.

**Risolti dopo la ri-esecuzione dei notebook:** l'ambiguità su quale esecuzione dell'ablazione fosse canonica (l'esecuzione divergente non esiste più; quella su disco è l'unica) e l'output mancante della tabella Rotated-MNIST in `mnist_check.ipynb`, ora presente e verificabile.

**Da valutare prima della consegna:** 4 degli 11 notebook (`har_adaptation_ablation`, `har_openset`, `loader`, `test_laplace_vs_mcmc`) hanno `execution_count` non sequenziali, cioè gli output salvati provengono da una sessione in cui altre celle erano già state eseguite, non da un run pulito dall'alto in basso. I numeri sono coerenti con tutto il resto e riproducibili, ma un revisore che aprisse quei notebook non vedrebbe una traccia di esecuzione lineare. `loader` e `test_laplace_vs_mcmc` costano pochi secondi da rieseguire; gli altri due sono più lenti.
