export const environment = {
  production: true,
  // Same-origin via nginx (Docker) or Angular proxy (ng serve).
  apiBaseUrl: '/api',
  wsBaseUrl: '', // resolved at runtime from window.location
};
