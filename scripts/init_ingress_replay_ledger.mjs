import {IngressReplayLedger} from './ingress_replay_ledger.mjs';

if (process.argv.length !== 4) throw new Error('usage: node init_ingress_replay_ledger.mjs HOST_LEDGER_FILE HOST_SUBJECT');
const ledger = new IngressReplayLedger(process.argv[2], process.argv[3], {initialize: true});
console.log(JSON.stringify({initialized: true, host_subject: ledger.subject}));
