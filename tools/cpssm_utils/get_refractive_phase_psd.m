function psd = get_refractive_phase_psd(varargin)
%GET_REFRACTIVE_PHASE_PSD Get refractive phase PSD bins and optional phase realization.

args = struct(varargin{:});

utils_dir = fileparts(mfilename('fullpath'));
cpssm_root = fullfile(fileparts(utils_dir), ...
    'submodules', 'gnss-scintillation-simulator');
addpath(genpath(fullfile(cpssm_root, 'libs')));

spectral_params = struct( ...
    'U', double(args.U), ...
    'p1', double(args.p1), ...
    'p2', double(args.p2), ...
    'mu0', double(args.mu0));

t_samp = double(args.t_samp);
rhof_veff_ratio = double(args.rhof_veff_ratio);
nfft = nicefftnum((double(args.sim_time) / t_samp) + 1);

frequency_hz = (-nfft/2 : nfft/2-1).' / (nfft * t_samp);
frequency_hz = frequency_hz(frequency_hz > 0);
sample_rate_hz = 1 / t_samp;

mu = 2 * pi * frequency_hz * rhof_veff_ratio;
P_mu = get_theorerical_phase_psd(mu, spectral_params);
S_phi_hz = rhof_veff_ratio * P_mu;

psd = struct();
psd.frequency_hz = frequency_hz;
psd.omega_rad_sample = 2 * pi * frequency_hz / sample_rate_hz;
psd.normalized_frequency_nyquist = frequency_hz / (sample_rate_hz / 2);
psd.mu = mu;
psd.P_mu = P_mu;
psd.S_phi_hz = S_phi_hz;
psd.rhof_veff_ratio = rhof_veff_ratio;
psd.sample_rate_hz = sample_rate_hz;
psd.sample_interval_s = t_samp;
psd.spectral_U = double(spectral_params.U);
psd.spectral_mu0 = double(spectral_params.mu0);
psd.spectral_p1 = double(spectral_params.p1);
psd.spectral_p2 = double(spectral_params.p2);
psd.nfft = nfft;

if isfield(args, 'seed') && ~isempty(args.seed)
    frequency_hz_full = (-nfft/2 : nfft/2-1).' / (nfft * t_samp);
    mu_full = 2 * pi * frequency_hz_full * rhof_veff_ratio;
    P_mu_full = get_theorerical_phase_psd(mu_full, spectral_params);
    D_mu = mu_full(2) - mu_full(1);

    phi_R = get_phase_realization(P_mu_full.', D_mu, nfft, double(args.seed));

    psd.frequency_hz_full = frequency_hz_full;
    psd.mu_full = mu_full;
    psd.P_mu_full = P_mu_full;
    psd.delta_mu = D_mu;
    psd.seed = double(args.seed);
    psd.phi_R = phi_R(:);
end

if strlength(string(args.output_file)) > 0
    output_file = char(args.output_file);
    output_dir = fileparts(output_file);
    if ~isempty(output_dir) && ~isfolder(output_dir)
        mkdir(output_dir);
    end
    save(output_file, '-struct', 'psd');
end

end
