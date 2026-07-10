% =============================================================================
% convert_fnirs_octave.m
% -----------------------------------------------------------------------------
% One-time converter for ds004022 fNIRS files.
%
% WHY: the fNIRS intensity matrix in this dataset is stored as
%      nirs_data.cnt.cnt.x = a MATLAB `table` (an MCOS object). Python readers
%      (scipy.io.loadmat / h5py) cannot deserialise MATLAB tables, so we convert
%      each `*_nirs.mat` to a plain-array `*_nirs_converted.mat` that
%      src/preprocess_fnirs.py loads directly.
%
% USAGE (MATLAB):   convert_fnirs_octave('data/ds004022')
% USAGE (CLI):      matlab -batch "convert_fnirs_octave('data/ds004022')"
%                   octave --no-gui tools/convert_fnirs_octave.m data/ds004022
%
% NOTE ON OCTAVE: core Octave does not read MATLAB `table` objects. Install the
% `tablicious` package (pkg install -forge tablicious; pkg load tablicious) or
% run this in MATLAB. If istable() is unavailable the script tries to coerce the
% object to a numeric matrix and warns if it cannot.
% =============================================================================

function convert_fnirs_octave(bids_root)
    if nargin < 1
        % allow `octave script.m <root>` invocation
        args = argv();
        if ~isempty(args), bids_root = args{1}; else bids_root = 'data/ds004022'; end
    end
    files = dir(fullfile(bids_root, 'sub-*', 'fnirs', '*_nirs.mat'));
    if isempty(files)
        error('No *_nirs.mat found under %s', bids_root);
    end
    fprintf('Found %d fNIRS files under %s\n', numel(files), bids_root);

    for i = 1:numel(files)
        in_path = fullfile(files(i).folder, files(i).name);
        out_path = strrep(in_path, '_nirs.mat', '_nirs_converted.mat');
        try
            S = load(in_path);
            nd = S.nirs_data;
            cnt = unwrap(nd.cnt, 'cnt');
            mrk = unwrap(nd.mrk, 'mrk');
            mnt = unwrap(nd.mnt, 'mnt');

            X = to_matrix(cnt.x);                 % (n_samples x n_channels)
            fs = double(cnt.fs);
            wavelengths = double(cnt.wavelengths(:))';
            clab = to_cellstr(cnt.clab);          % channel labels 'S1_D1 760' ...
            mrk_pos = double(mrk.pos(:))';        % marker sample indices
            mrk_toe = double(mrk.toe(:))';        % marker type codes
            mrk_fs  = double(mrk.fs);

            save(out_path, 'X', 'fs', 'wavelengths', 'clab', ...
                 'mrk_pos', 'mrk_toe', 'mrk_fs', '-v7');
            fprintf('  [ok] %s  (X = %d x %d)\n', files(i).name, size(X,1), size(X,2));
        catch err
            fprintf(2, '  [skip] %s : %s\n', files(i).name, err.message);
        end
    end
end

function s = unwrap(s, name)
    % ds004022 double-wraps: nirs_data.cnt.cnt, nirs_data.mrk.mrk, ...
    if isstruct(s) && isfield(s, name)
        s = s.(name);
    end
end

function M = to_matrix(x)
    if istable(x)
        M = table2array(x);
    elseif isnumeric(x)
        M = x;
    else
        % last resort: try struct2array / double
        try
            M = double(x);
        catch
            error('cnt.x is a %s that could not be converted to a matrix', class(x));
        end
    end
    M = double(M);
end

function c = to_cellstr(x)
    if iscell(x)
        c = cellfun(@char, x, 'UniformOutput', false);
    elseif ischar(x)
        c = cellstr(x);
    else
        c = {};
    end
end
