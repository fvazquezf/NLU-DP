import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Input
from algorithm import ArcEager, Transition, Sample
from state import State
from conllu_token import Token

class ParserMLP:
    """
    A Multi-Layer Perceptron (MLP) class for a dependency parser, using TensorFlow and Keras.

    This class implements a neural network model designed to predict transitions in a dependency 
    parser. It utilizes the Keras Functional API, which is more suited for multi-task learning scenarios 
    like this one. The network is trained to map parsing states to transition actions, facilitating 
    the parsing process in natural language processing tasks.

    Attributes:
        word_emb_dim (int): Dimensionality of the word embeddings. Defaults to 100.
        hidden_dim (int): Dimension of the hidden layer in the neural network. Defaults to 64.
        epochs (int): Number of training epochs. Defaults to 1.
        batch_size (int): Size of the batches used in training. Defaults to 64.

    Methods:
        train(training_samples, dev_samples): Trains the MLP model using the provided training and 
            development samples. It maps these samples to IDs that can be processed by an embedding 
            layer and then calls the Keras compile and fit functions.

        evaluate(samples): Evaluates the performance of the model on a given set of samples. The 
            method aims to assess the accuracy in predicting both the transition and dependency types, 
            with expected accuracies ranging between 75% and 85%.

        run(sents): Processes a list of sentences (tokens) using the trained model to perform dependency 
            parsing. This method implements the vertical processing of sentences to predict parser 
            transitions for each token.

        Feel free to add other parameters and functions you might need to create your model
    """

    def __init__(self, word_emb_dim: int = 100, hidden_dim: int = 64, 
                 epochs: int = 1, batch_size: int = 64):
        """
        Initializes the ParserMLP class with the specified dimensions and training parameters.

        Parameters:
            word_emb_dim (int): The dimensionality of the word embeddings.
            hidden_dim (int): The size of the hidden layer in the MLP.
            epochs (int): The number of epochs for training the model.
            batch_size (int): The batch size used during model training.
        """
        self.word_emb_dim = word_emb_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.model = None
        self.arc_eager = ArcEager()
        
        # Mappings
        self.word2id = {"<PAD>": 0, "<UNK>": 1}
        self.upos2id = {"<PAD>": 0, "<UNK>": 1}
        self.action2id = {}
        self.deprel2id = {}
        self.id2action = []
        self.id2deprel = []

    def _build_vocab(self, samples: list['Sample']):
        """Helper to build vocabularies from training samples."""
        actions = set()
        deprels = set()
        
        for sample in samples:
            # Features are strings: [w_stk..., w_buf..., u_stk..., u_buf...]
            feats = sample.state_to_feats()
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
            
            # Outputs
            t = sample.transition
            actions.add(t.action)
            if t.dependency:
                deprels.add(t.dependency)
        
        # Create mappings for outputs
        self.id2action = sorted(list(actions))
        self.action2id = {a: i for i, a in enumerate(self.id2action)}
        
        self.id2deprel = sorted(list(deprels))
        self.deprel2id = {d: i for i, d in enumerate(self.id2deprel)}
        
        print(f"Vocab built: {len(self.word2id)} words, {len(self.upos2id)} UPOS tags.")
        print(f"Outputs: {len(self.action2id)} actions, {len(self.deprel2id)} dependency labels.")

    def _vectorize(self, samples: list['Sample']):
        """Converts Sample objects into numeric arrays for inputs and outputs."""
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
            
            # Map words/upos to IDs
            w_ids = [self.word2id.get(w, self.word2id["<UNK>"]) for w in words]
            u_ids = [self.upos2id.get(u, self.upos2id["<UNK>"]) for u in upos]
            
            X_words.append(w_ids)
            X_upos.append(u_ids)
            
            # Map outputs
            t = sample.transition
            Y_actions.append(self.action2id[t.action])
            
            # For Deprel, if None (e.g. SHIFT/REDUCE), we can use a dummy index or handle differently.
            # Here we just use 0 or a specific 'None' class if it existed, but usually 
            # the loss is masked or we just predict a dummy label that is ignored.
            # For simplicity, we will map None to index 0 (assuming 'root' or first label is 0) 
            # but in inference we ignore deprel for SHIFT/REDUCE.
            if t.dependency:
                Y_deprels.append(self.deprel2id.get(t.dependency, 0))
            else:
                Y_deprels.append(0) # Placeholder
                
        return (np.array(X_words), np.array(X_upos)), (np.array(Y_actions), np.array(Y_deprels))

    def train(self, training_samples: list['Sample'], dev_samples: list['Sample']):
        """
        Trains the MLP model using the provided training and development samples.

        This method prepares the training data by mapping samples to IDs suitable for 
        embedding layers and then proceeds to compile and fit the Keras model.

        Parameters:
            training_samples (list[Sample]): A list of training samples for the parser.
            dev_samples (list[Sample]): A list of development samples used for model validation.
        """
        # 1. Build Vocabulary
        print("Building vocabulary...")
        self._build_vocab(training_samples)
        
        # 2. Vectorize Data
        print("Vectorizing data...")
        X_train, Y_train = self._vectorize(training_samples)
        X_dev, Y_dev = self._vectorize(dev_samples)
        
        # 3. Define Model Architecture (Functional API)
        # Inputs
        input_words = Input(shape=(4,), name='words_input') # 2 stack + 2 buffer = 4
        input_upos = Input(shape=(4,), name='upos_input')
        
        # Embeddings
        emb_words = layers.Embedding(input_dim=len(self.word2id), output_dim=self.word_emb_dim)(input_words)
        emb_upos = layers.Embedding(input_dim=len(self.upos2id), output_dim=50)(input_upos) # Smaller dim for UPOS
        
        # Flatten and Concatenate
        flat_words = layers.Flatten()(emb_words)
        flat_upos = layers.Flatten()(emb_upos)
        concat = layers.Concatenate()([flat_words, flat_upos])
        
        # Hidden Layer
        hidden = layers.Dense(self.hidden_dim, activation='relu')(concat)
        dropout = layers.Dropout(0.2)(hidden)
        
        # Outputs
        out_action = layers.Dense(len(self.action2id), activation='softmax', name='action_output')(dropout)
        out_deprel = layers.Dense(len(self.deprel2id), activation='softmax', name='deprel_output')(dropout)
        
        self.model = models.Model(inputs=[input_words, input_upos], outputs=[out_action, out_deprel])
        
        self.model.compile(
            optimizer='adam',
            loss={'action_output': 'sparse_categorical_crossentropy', 'deprel_output': 'sparse_categorical_crossentropy'},
            loss_weights={'action_output': 1.0, 'deprel_output': 0.5}, # Weight action loss higher
            metrics=['accuracy']
        )
        
        self.model.summary()
        
        # 4. Train
        self.model.fit(
            x=X_train, 
            y={'action_output': Y_train[0], 'deprel_output': Y_train[1]},
            validation_data=(X_dev, {'action_output': Y_dev[0], 'deprel_output': Y_dev[1]}),
            epochs=self.epochs,
            batch_size=self.batch_size
        )

    def evaluate(self, samples: list['Sample']):
        """
        Evaluates the model's performance on a set of samples.

        This method is used to assess the accuracy of the model in predicting the correct
        transition and dependency types. The expected accuracy range is between 75% and 85%.

        Parameters:
            samples (list[Sample]): A list of samples to evaluate the model's performance.
        """
        X, Y = self._vectorize(samples)
        results = self.model.evaluate(X, {'action_output': Y[0], 'deprel_output': Y[1]}, verbose=0)
        
        print(f"Evaluation Results:")
        print(f"Total Loss: {results[0]:.4f}")
        print(f"Action Accuracy: {results[3]:.4f}") # Index depends on metrics order, usually loss, act_loss, dep_loss, act_acc, dep_acc
        print(f"Deprel Accuracy: {results[4]:.4f}")

    def run(self, sents: list[list[Token]]):
        """
        Executes the model on a list of sentences to perform dependency parsing.

        This method implements the vertical processing of sentences, predicting parser 
        transitions for each token in the sentences.

        Parameters:
            sents (list[Token]): A list of sentences, where each sentence is represented 
                                 as a list of Token objects.
        """
        states = [self.arc_eager.create_initial_state(sent) for sent in sents]
        
        # Keep track of indices that are still active
        active_indices = list(range(len(states)))
        
        while active_indices:
            # Prepare batch
            batch_states = [states[i] for i in active_indices]
            dummy_samples = [Sample(s, None) for s in batch_states]
            
            # Vectorize
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
            
            # Predict
            preds = self.model.predict(X_in, verbose=0)
            pred_actions = preds[0]
            pred_deprels = preds[1]
            
            next_active_indices = []
            
            for idx_in_batch, sent_idx in enumerate(active_indices):
                state = states[sent_idx]
                
                # Get best valid action
                action_probs = pred_actions[idx_in_batch]
                sorted_action_indices = np.argsort(action_probs)[::-1]
                
                best_deprel = self.id2deprel[np.argmax(pred_deprels[idx_in_batch])]
                
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
                
                # Check if finished
                if not self.arc_eager.final_state(state) and valid_found:
                    next_active_indices.append(sent_idx)
            
            active_indices = next_active_indices

        # Apply Arcs to Tokens
        for i, sent in enumerate(sents):
            state = states[i]
            # Arcs are (head, dep, dependent_id)
            # We need to map these to the token objects
            # Token.id matches dependent_id
            
            # Create a lookup for tokens by ID
            id2token = {tok.id: tok for tok in sent}
            
            for (head_id, dep_label, child_id) in state.A:
                if child_id in id2token:
                    id2token[child_id].head = head_id
                    id2token[child_id].dep = dep_label
                    
        return sents
    
# Main Steps for Processing Sentences:
# 1. Initialize: Create the initial state for each sentence.
# 2. Feature Representation: Convert states to their corresponding list of features.
# 3. Model Prediction: Use the model to predict the next transition and dependency type for all current states.
# 4. Transition Sorting: For each prediction, sort the transitions by likelihood using numpy.argsort, 
#    and select the most likely dependency type with argmax.
# 5. Validation Check: Verify if the selected transition is valid for each prediction. If not, select the next most likely one.
# 6. State Update: Apply the selected actions to update all states, and create a list of new states.
# 7. Final State Check: Remove sentences that have reached a final state.
# 8. Iterative Process: Repeat steps 2 to 7 until all sentences have reached their final state.

if __name__ == "__main__":
    
    model = ParserMLP()