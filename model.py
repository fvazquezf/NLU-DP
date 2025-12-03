import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Input
import pickle
import os
from algorithm import ArcEager, Transition, Sample
from conllu_token import Token

class ParserMLP:
    """
    Improved Multi-Layer Perceptron (MLP) for dependency parsing.
    
    Architecture based on standard transition-based parsing models:
    Inputs -> Embeddings -> Flatten & Concat -> Hidden Layers -> Output Heads
    """

    def __init__(self, word_emb_dim: int = 100, hidden_dim: int = 256, 
                 epochs: int = 5, batch_size: int = 32, learning_rate: float = 0.001):
        """
        Initializes the model with configurable hyperparameters.
        """
        self.word_emb_dim = word_emb_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.model = None
        self.arc_eager = ArcEager()
        
        # Mappings
        self.word2id = {"<PAD>": 0, "<UNK>": 1}
        self.upos2id = {"<PAD>": 0, "<UNK>": 1}
        self.action2id = {}
        self.deprel2id = {}
        self.id2action = []
        self.id2deprel = []
        
        # Input shapes (detected during training)
        self.n_word_feats = 0
        self.n_upos_feats = 0

    def _build_vocab(self, samples: list['Sample']):
        """Builds vocabulary from training samples."""
        actions = set()
        deprels = set()
        
        for sample in samples:
            feats = sample.state_to_feats()
            # Dynamic feature length detection
            num_feats = len(feats)
            mid = num_feats // 2
            
            words = feats[:mid]
            upos = feats[mid:]
            
            for w in words:
                if w not in self.word2id:
                    self.word2id[w] = len(self.word2id)
            for u in upos:
                if u not in self.upos2id:
                    self.upos2id[u] = len(self.upos2id)
            
            t = sample.transition
            actions.add(t.action)
            if t.dependency:
                deprels.add(t.dependency)
        
        self.id2action = sorted(list(actions))
        self.action2id = {a: i for i, a in enumerate(self.id2action)}
        
        self.id2deprel = sorted(list(deprels))
        self.deprel2id = {d: i for i, d in enumerate(self.id2deprel)}
        
        print(f"Vocab built: {len(self.word2id)} words, {len(self.upos2id)} UPOS tags.")

    def _vectorize(self, samples: list['Sample']):
        """Converts Samples to numpy arrays."""
        X_words = []
        X_upos = []
        Y_actions = []
        Y_deprels = []
        
        for sample in samples:
            feats = sample.state_to_feats()
            num_feats = len(feats)
            mid = num_feats // 2
            words = feats[:mid]
            upos = feats[mid:]
            
            # Use dictionary .get with default for UNK handling
            w_ids = [self.word2id.get(w, self.word2id["<UNK>"]) for w in words]
            u_ids = [self.upos2id.get(u, self.upos2id["<UNK>"]) for u in upos]
            
            X_words.append(w_ids)
            X_upos.append(u_ids)
            
            t = sample.transition
            Y_actions.append(self.action2id[t.action])
            Y_deprels.append(self.deprel2id.get(t.dependency, 0))
                
        return (np.array(X_words), np.array(X_upos)), (np.array(Y_actions), np.array(Y_deprels))

    def build_model(self):
        """Constructs the Keras Functional model dynamically."""
        # Inputs
        input_words = Input(shape=(self.n_word_feats,), name='words_input')
        input_upos = Input(shape=(self.n_upos_feats,), name='upos_input')
        
        # Embeddings [cite: 717]
        emb_words = layers.Embedding(input_dim=len(self.word2id), output_dim=self.word_emb_dim)(input_words)
        emb_upos = layers.Embedding(input_dim=len(self.upos2id), output_dim=50)(input_upos)
        
        # Flatten and Concatenate
        flat_words = layers.Flatten()(emb_words)
        flat_upos = layers.Flatten()(emb_upos)
        x = layers.Concatenate()([flat_words, flat_upos])
        
        # Hidden Layers [cite: 720]
        # First dense layer
        x = layers.Dense(self.hidden_dim, activation='relu')(x)
        x = layers.Dropout(0.3)(x) # Dropout to prevent overfitting
        
        # Second dense layer (optional but helps with complex patterns)
        x = layers.Dense(self.hidden_dim // 2, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output Heads [cite: 851, 852]
        out_action = layers.Dense(len(self.action2id), activation='softmax', name='action_output')(x)
        out_deprel = layers.Dense(len(self.deprel2id), activation='softmax', name='deprel_output')(x)
        
        self.model = models.Model(inputs=[input_words, input_upos], outputs=[out_action, out_deprel])
        
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss={'action_output': 'sparse_categorical_crossentropy', 'deprel_output': 'sparse_categorical_crossentropy'},
            loss_weights={'action_output': 1.0, 'deprel_output': 0.75}, # Tweaked weight
            metrics=['accuracy']
        )
        self.model.summary()

    def train(self, training_samples: list['Sample'], dev_samples: list['Sample']):
        # 1. Build Vocab
        print("Building vocabulary...")
        self._build_vocab(training_samples)
        
        # 2. Detect input shapes from the first sample
        dummy_feats = training_samples[0].state_to_feats()
        total_feats = len(dummy_feats)
        self.n_word_feats = total_feats // 2
        self.n_upos_feats = total_feats // 2
        print(f"Detected Feature Shape: {self.n_word_feats} words, {self.n_upos_feats} tags.")
        
        # 3. Vectorize
        print("Vectorizing data...")
        X_train, Y_train = self._vectorize(training_samples)
        X_dev, Y_dev = self._vectorize(dev_samples)
        
        # 4. Build and Train
        self.build_model()
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=2, restore_best_weights=True
        )
        
        history = self.model.fit(
            x=X_train, 
            y={'action_output': Y_train[0], 'deprel_output': Y_train[1]},
            validation_data=(X_dev, {'action_output': Y_dev[0], 'deprel_output': Y_dev[1]}),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping]
        )
        return history

    def evaluate(self, samples: list['Sample']):
        X, Y = self._vectorize(samples)
        results = self.model.evaluate(X, {'action_output': Y[0], 'deprel_output': Y[1]}, verbose=1)
        return results

    def run(self, sents: list[list[Token]]):
        """Vertical processing for efficient inference."""
        states = [self.arc_eager.create_initial_state(sent) for sent in sents]
        active_indices = list(range(len(states)))
        
        while active_indices:
            batch_states = [states[i] for i in active_indices]
            # Create dummy samples to use state_to_feats
            dummy_samples = [Sample(s, Transition("SHIFT")) for s in batch_states]
            
            # Vectorize on the fly
            X_words = []
            X_upos = []
            for samp in dummy_samples:
                feats = samp.state_to_feats()
                mid = len(feats) // 2
                w_ids = [self.word2id.get(w, self.word2id["<UNK>"]) for w in feats[:mid]]
                u_ids = [self.upos2id.get(u, self.upos2id["<UNK>"]) for u in feats[mid:]]
                X_words.append(w_ids)
                X_upos.append(u_ids)
            
            X_in = [np.array(X_words), np.array(X_upos)]
            preds = self.model.predict(X_in, verbose=0)
            pred_actions = preds[0]
            pred_deprels = preds[1]
            
            next_active_indices = []
            
            for idx_in_batch, sent_idx in enumerate(active_indices):
                state = states[sent_idx]
                action_probs = pred_actions[idx_in_batch]
                sorted_action_indices = np.argsort(action_probs)[::-1]
                
                best_deprel_idx = np.argmax(pred_deprels[idx_in_batch])
                best_deprel = self.id2deprel[best_deprel_idx]
                
                valid_found = False
                for act_id in sorted_action_indices:
                    act_str = self.id2action[act_id]
                    is_valid = False
                    if act_str == ArcEager.SHIFT:
                        if len(state.B) > 0: is_valid = True
                    elif act_str == ArcEager.REDUCE:
                        is_valid = self.arc_eager.REDUCE_is_valid(state)
                    elif act_str == ArcEager.LA:
                        is_valid = self.arc_eager.LA_is_valid(state)
                    elif act_str == ArcEager.RA:
                        is_valid = self.arc_eager.RA_is_valid(state)
                    
                    if is_valid:
                        self.arc_eager.apply_transition(state, Transition(act_str, best_deprel))
                        valid_found = True
                        break
                
                if not self.arc_eager.final_state(state) and valid_found:
                    next_active_indices.append(sent_idx)
            
            active_indices = next_active_indices

        # Apply arcs to Tokens
        for i, sent in enumerate(sents):
            state = states[i]
            id2token = {tok.id: tok for tok in sent}
            for (head_id, dep_label, child_id) in state.A:
                if child_id in id2token:
                    id2token[child_id].head = head_id
                    id2token[child_id].dep = dep_label
        return sents

    def save_model(self, path="parser_model"):
        """Saves model weights and vocabulary."""
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Save Keras model
        self.model.save(os.path.join(path, "keras_model.keras"))
        
        # Save metadata (vocab, IDs)
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump({
                "word2id": self.word2id,
                "upos2id": self.upos2id,
                "action2id": self.action2id,
                "deprel2id": self.deprel2id,
                "id2action": self.id2action,
                "id2deprel": self.id2deprel,
                "n_word_feats": self.n_word_feats,
                "n_upos_feats": self.n_upos_feats
            }, f)
        print(f"Model saved to {path}")

    def load_model(self, path="parser_model"):
        """Loads a saved model."""
        # Load Metadata
        with open(os.path.join(path, "metadata.pkl"), "rb") as f:
            data = pickle.load(f)
            self.word2id = data["word2id"]
            self.upos2id = data["upos2id"]
            self.action2id = data["action2id"]
            self.deprel2id = data["deprel2id"]
            self.id2action = data["id2action"]
            self.id2deprel = data["id2deprel"]
            self.n_word_feats = data.get("n_word_feats", 2)
            self.n_upos_feats = data.get("n_upos_feats", 2)
            
        # Load Keras Model
        self.model = models.load_model(os.path.join(path, "keras_model.keras"))
        print(f"Model loaded from {path}")