// Initialisierung des Canvas und des Spielfelds
const canvas = document.getElementById('tetris');
const context = canvas.getContext('2d');

context.scale(20, 20);

// Farben und Tetromino-Formen
const colors = [
    null,
    '#FF0D72', // T
    '#0DC2FF', // O
    '#0DFF72', // L
    '#F538FF', // J
    '#FF8E0D', // I
    '#FFE138', // S
    '#3877FF', // Z
];

const tetrominoes = 'ILJOTSZ';

// Arena und Spieler
const arena = createMatrix(12, 20);

const player = {
    pos: { x: 0, y: 0 },
    matrix: null,
    score: 0,
};

let clearedLines = 0; // Variable zum Zählen der gelöschten Reihen

// KI-Parameter
let model;
const memory = [];
const maxMemory = 5000;
const batchSize = 64;
let epsilon = 1.0; // Exploitation vs. Exploration
const epsilonMin = 0.01;
const epsilonDecay = 0.995;
const gamma = 0.95; // Discount-Faktor

// Modell erstellen
async function createModel() {
    model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [258], units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 4, activation: 'linear' })); // Links, Rechts, Drehen, Fallenlassen
    model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
}

// Zustand des Spielfelds abrufen
function getState() {
    const state = [];
    // Spielerposition normalisieren
    state.push(player.pos.x / arena[0].length);
    state.push(player.pos.y / arena.length);
    // Spielfeldzustand
    arena.forEach(row => row.forEach(value => state.push(value ? 1 : 0)));
    // Aktuelles Tetromino als 4x4-Matrix
    const tetrominoMatrix = padMatrix(player.matrix, 4, 4);
    tetrominoMatrix.forEach(row => row.forEach(value => state.push(value ? 1 : 0)));
    return state;
}

// Hilfsfunktion zum Auffüllen der Matrix auf 4x4
function padMatrix(matrix, height, width) {
    const newMatrix = [];
    for (let y = 0; y < height; y++) {
        const row = [];
        for (let x = 0; x < width; x++) {
            if (y < matrix.length && x < matrix[y].length) {
                row.push(matrix[y][x]);
            } else {
                row.push(0);
            }
        }
        newMatrix.push(row);
    }
    return newMatrix;
}

// Aktion basierend auf dem aktuellen Zustand wählen
async function chooseAction(state) {
    if (Math.random() < epsilon) {
        // Zufällige Aktion (Exploration)
        return Math.floor(Math.random() * 4);
    } else {
        // Beste Aktion basierend auf Vorhersage (Exploitation)
        const input = tf.tensor2d([state]);
        const prediction = model.predict(input);
        const action = prediction.argMax(1).dataSync()[0];
        tf.dispose([input, prediction]);
        return action;
    }
}

// Erfahrung zur Erinnerung hinzufügen
function remember(state, action, reward, nextState, done) {
    if (memory.length >= maxMemory) {
        memory.shift();
    }
    memory.push({ state, action, reward, nextState, done });
}

// Modell trainieren
async function trainModel() {
    if (memory.length < batchSize) return;

    const batch = [];
    for (let i = 0; i < batchSize; i++) {
        batch.push(memory[Math.floor(Math.random() * memory.length)]);
    }

    const states = batch.map(m => m.state);
    const nextStates = batch.map(m => m.nextState);

    const statesTensor = tf.tensor2d(states);
    const nextStatesTensor = tf.tensor2d(nextStates);

    const qValues = model.predict(statesTensor);
    const qValuesNext = model.predict(nextStatesTensor);

    const targetQs = qValues.arraySync();

    const qValuesNextArray = qValuesNext.arraySync();

    batch.forEach((memoryItem, index) => {
        if (memoryItem.done) {
            targetQs[index][memoryItem.action] = memoryItem.reward;
        } else {
            const maxQ = Math.max(...qValuesNextArray[index]);
            targetQs[index][memoryItem.action] = memoryItem.reward + gamma * maxQ;
        }
    });

    const targetQsTensor = tf.tensor2d(targetQs);

    await model.fit(statesTensor, targetQsTensor, { epochs: 1, verbose: 0 });

    tf.dispose([statesTensor, nextStatesTensor, qValues, qValuesNext, targetQsTensor]);

    if (epsilon > epsilonMin) {
        epsilon *= epsilonDecay;
    }
}

// Hauptspiel-Loop mit KI-Steuerung
async function update(time = 0) {
    // Zustand vor der Aktion
    const state = getState();

    // Aktion wählen
    const action = await chooseAction(state);

    // Aktion ausführen
    switch (action) {
        case 0:
            playerMove(-1); // Links
            break;
        case 1:
            playerMove(1); // Rechts
            break;
        case 2:
            playerRotate(1); // Drehen
            break;
        case 3:
            playerDrop(); // Fallenlassen
            break;
    }

    // Zustand nach der Aktion
    const nextState = getState();

    // Belohnung berechnen
    const reward = calculateReward();

    // Prüfen, ob das Spiel vorbei ist
    const done = isGameOver();

    // Erfahrung speichern
    remember(state, action, reward, nextState, done);

    // Modell trainieren
    await trainModel();

    // Spielfeld zeichnen
    draw();

    if (!done) {
        requestAnimationFrame(update);
    } else {
        // Spiel zurücksetzen
        playerReset();
        arena.forEach(row => row.fill(0));
        requestAnimationFrame(update);
    }
}

// Rest des Codes (Spielmechanik, Zeichnen, Tetromino-Generierung, etc.)

function createMatrix(w, h) {
    const matrix = [];
    while (h--) {
        matrix.push(new Array(w).fill(0));
    }
    return matrix;
}

function createPiece(type) {
    switch (type) {
        case 'T':
            return [
                [0, 0, 0],
                [1, 1, 1],
                [0, 1, 0],
            ];
        case 'O':
            return [
                [2, 2],
                [2, 2],
            ];
        case 'L':
            return [
                [0, 3, 0],
                [0, 3, 0],
                [0, 3, 3],
            ];
        case 'J':
            return [
                [0, 4, 0],
                [0, 4, 0],
                [4, 4, 0],
            ];
        case 'I':
            return [
                [0, 0, 5, 0],
                [0, 0, 5, 0],
                [0, 0, 5, 0],
                [0, 0, 5, 0],
            ];
        case 'S':
            return [
                [0, 6, 6],
                [6, 6, 0],
                [0, 0, 0],
            ];
        case 'Z':
            return [
                [7, 7, 0],
                [0, 7, 7],
                [0, 0, 0],
            ];
        default:
            // Falls ein unbekannter Typ übergeben wird, geben wir eine Standardform zurück
            return [
                [0, 0, 0],
                [1, 1, 1],
                [0, 1, 0],
            ];
    }
}

function drawMatrix(matrix, offset) {
    matrix.forEach((row, y) => {
        row.forEach((value, x) => {
            if (value !== 0) {
                context.fillStyle = colors[value];
                context.fillRect(x + offset.x, y + offset.y, 1, 1);
            }
        });
    });
}

function merge(arena, player) {
    player.matrix.forEach((row, y) => {
        row.forEach((value, x) => {
            if (value !== 0) {
                arena[y + player.pos.y][x + player.pos.x] = value;
            }
        });
    });
}

function rotate(matrix, dir) {
    // Transponieren der Matrix
    for (let y = 0; y < matrix.length; ++y) {
        for (let x = 0; x < y; ++x) {
            [
                matrix[x][y],
                matrix[y][x],
            ] = [
                matrix[y][x],
                matrix[x][y],
            ];
        }
    }
    // Reihenfolge umkehren
    if (dir > 0) {
        matrix.forEach(row => row.reverse());
    } else {
        matrix.reverse();
    }
}

function collide(arena, player) {
    const m = player.matrix;
    const o = player.pos;
    for (let y = 0; y < m.length; ++y) {
        for (let x = 0; x < m[y].length; ++x) {
            if (m[y][x] !== 0 &&
                (arena[y + o.y] &&
                 arena[y + o.y][x + o.x]) !== 0) {
                return true;
            }
        }
    }
    return false;
}

function arenaSweep() {
    let rowCount = 0;
    outer: for (let y = arena.length -1; y >= 0; --y) {
        for (let x = 0; x < arena[y].length; ++x) {
            if (arena[y][x] === 0) {
                continue outer;
            }
        }

        const row = arena.splice(y, 1)[0].fill(0);
        arena.unshift(row);
        ++y;
        rowCount++;
    }
    clearedLines = rowCount; // Aktualisiere die Anzahl der gelöschten Reihen
    player.score += rowCount * 10;
}

function draw() {
    context.fillStyle = '#000';
    context.fillRect(0, 0, canvas.width, canvas.height);

    drawMatrix(arena, { x: 0, y: 0 });
    drawMatrix(player.matrix, player.pos);
}

function playerMove(dir) {
    player.pos.x += dir;
    if (collide(arena, player)) {
        player.pos.x -= dir;
    }
}

function playerRotate(dir) {
    const pos = player.pos.x;
    let offset = 1;
    rotate(player.matrix, dir);
    while (collide(arena, player)) {
        player.pos.x += offset;
        offset = -(offset + (offset > 0 ? 1 : -1));
        if (offset > player.matrix[0].length) {
            rotate(player.matrix, -dir);
            player.pos.x = pos;
            return;
        }
    }
}

function playerDrop() {
    player.pos.y++;
    if (collide(arena, player)) {
        player.pos.y--;
        merge(arena, player);
        arenaSweep();
        playerReset();
    }
}

function playerReset() {
    player.matrix = createPiece(tetrominoes[Math.floor(Math.random() * tetrominoes.length)]);
    player.pos.y = 0;
    player.pos.x = (arena[0].length / 2 | 0) -
        (player.matrix[0].length / 2 | 0);
    if (collide(arena, player)) {
        // Spielende
        player.score = 0;
        arena.forEach(row => row.fill(0));
    }
}

function calculateReward() {
    // Einfache Belohnungsfunktion
    let reward = 0;
    // Belohnung für gelöschte Reihen
    reward += clearedLines * 10;
    clearedLines = 0; // Zurücksetzen für den nächsten Zug
    // Bestrafung für jeden Zug, um die KI zu motivieren, effizient zu sein
    reward -= 0.1;
    return reward;
}

function isGameOver() {
    // Überprüfen, ob das Spielfeld voll ist (wenn ein Tetromino beim Erzeugen kollidiert)
    return collide(arena, player);
}

// Starten des Spiels und der KI
async function startGame() {
    await createModel();
    playerReset();
    update();
}

startGame();
