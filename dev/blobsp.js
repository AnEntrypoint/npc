import { BLOB } from '../src/games/blob.js';

export const BLOBSP = Object.assign({}, BLOB, { id: 'blobsp', name: 'Blob self-play (dev)', learners: BLOB.agents });
