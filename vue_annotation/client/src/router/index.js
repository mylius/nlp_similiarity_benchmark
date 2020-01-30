import Vue from 'vue'
import Router from 'vue-router'
import Sentences from '@/components/Sentences'
import addsentence from '@/components/AddSentence'
import editsentence from '@/components/EditSentence'

Vue.use(Router)

export default new Router({
  mode: 'history',
  routes: [
    {
      path: '/',
      name: 'Sentences',
      component: Sentences
    },
    {
      path: '/sentences/add',
      component: addsentence,
      name: 'addsentence'
    },
    {
      path: '/sentences/:id/edit',
      component: editsentence,
      name: 'editsentence'
    }
  ]
})