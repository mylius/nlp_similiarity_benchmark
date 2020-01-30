<template>
<div class="sentences">
    <div>
    <h3 >Reference sentence:</h3>
    <p>{{sentences[ids[0]].sent}}</p>
    </div>
    <div class="comp">
    <div>
    <h3>Sentence 1:</h3>
    <p>{{sentences[ids[1]].sent}}</p>
    </div>
    <div>
    <h3>Sentence 2:</h3>
    <p>{{sentences[ids[2]].sent}}&nbsp;</p>
    </div>
    </div>
    &nbsp;
    &nbsp;
    Which sentence is more similar to the reference sentence?
    &nbsp;  
    <div>
    <button v-on:click="select1">1</button>
    <button v-on:click="select2">2</button>
    <button v-on:click="skip">skip</button>
    </div>
</div>
</template>

<script>
import DataService from '@/services/dataservice';
export default {
    name: 'annotation',
    data(){
        return {
            sentences: [],
            error: String,
            ids: [],
        }
    },
    async created() {
        try{
            this.sentences = await DataService.getSentences();
            this.setIds();
        } catch(err) {
            this.error = err.message;
        }
        
    },
    methods: {
        async updateSentence () {
            await DataService.updateSentence({
                id:  this.sentences[this.ids[0]].id,
                sent: this.sentences[this.ids[0]].sent,
                annotations: this.sentences[this.ids[0]].annotations
            })
        },
        /**
         * A simple randmon choice with no replacing algorithm.
         * @param {Array} InArray The array to be chosen from.
         * @param {Number} len The number of values to be chosen.
         * @return {Array}
         */
        randomChoiceNoReplace(InArray, len) {
            var bucket = [];
            var OutArray = new Array(len).fill(null);
            var randomIndex = 0;

            for (let i in OutArray) {
            bucket.push(InArray[i]);
            }
            for (let entry in OutArray) {
            randomIndex = Math.floor(Math.random() * bucket.length);
            OutArray[entry] = bucket.splice(randomIndex, 1)[0];
            }
            return OutArray
        },
        /**
         * Randomly selects the ids for the 3 sentences.
         * @return {Array} An array of 3 numbers.
         */
        setIds() {
            this.ids = this.randomChoiceNoReplace([...Array(this.sentences.length).keys()], 3);
        },
        /**
         * The actions to be done when selecting button 1. 
         * Creates the annotation, updates the sentence annotation in the mongodb and choses 3 new sentences.
         */
        select1() {
            this.annotate(1,2); 
            this.updateSentence();
            this.ids = this.randomChoiceNoReplace([...Array(this.sentences.length).keys()], 3);
            },
        /**
         * The actions to be done when selecting button 2. 
         * Creates the annotation, updates the sentence annotation in the mongodb and choses 3 new sentences.
         */
        select2() {
            
            this.annotate(2,1);
            this.updateSentence();
            this.ids = this.randomChoiceNoReplace([...Array(this.sentences.length).keys()], 3);
        },
        /**
         * The actions to be done when selecting the skip button. 
         * Choses 3 new sentences.
         */
        skip() {
            this.ids = this.randomChoiceNoReplace([...Array(this.sentences.length).keys()], 3);
        },
        /**
         * Annotates the reference sentences:
         * Writes a dict into .annotations of the reference sentence with "(a,b)" being the key and increasing the value by one.
         * @param {number} a The id of the the sentence more similar to the reference.
         * @param {number} b The id of the the sentence less similar to the reference.
         */
        annotate(a,b) {
            if (this.sentences[this.ids[0]].annotations["("+this.ids[a]+","+this.ids[b]+")"] == null){
               this.sentences[this.ids[0]].annotations["("+this.ids[a]+","+this.ids[b]+")"] = 1
            } else{
                this.sentences[this.ids[0]].annotations["("+this.ids[a]+","+this.ids[b]+")"] += 1;
            }
        }
    }
}

</script>
<style scoped>
.sentences {
    display: grid;
    grid-template-columns: repeat(1,1fr);
    grid-auto-rows: minmax(50px, auto);
    margin-left:10%;
    margin-right:10%;
}
.comp {
    display: grid;
    grid-template-columns: repeat(2,1fr);
    grid-column-gap: 20px;
    grid-auto-rows: minmax(50px, auto);
}
</style>